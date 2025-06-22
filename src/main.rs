use std::collections::HashMap;
use std::fs::{self, File, Metadata};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use regex::Regex;
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Metadata for a single file discovered during indexing.
/// Includes timestamps, permissions and optional checksum and preview.
pub struct FileRecord {
    pub path: PathBuf,
    pub name: String,
    pub size: u64,
    pub created: Option<u64>,
    pub modified: u64,
    pub accessed: Option<u64>,
    pub permissions: String,
    pub is_dir: bool,
    pub extension: Option<String>,
    pub checksum: Option<String>,
    pub content_preview: Option<String>,
}

#[derive(Debug, Clone)]
/// Options controlling how directories are traversed and files are indexed.
pub struct IndexConfig {
    pub max_depth: Option<usize>,
    pub include_content: bool,
    pub content_preview_size: usize,
    pub compute_checksums: bool,
    pub follow_symlinks: bool,
    pub ignore_patterns: Vec<Regex>,
    pub include_patterns: Vec<Regex>,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            include_content: false,
            content_preview_size: 1024,
            compute_checksums: false,
            follow_symlinks: false,
            ignore_patterns: vec![
                Regex::new(r"\.git").unwrap(),
                Regex::new(r"node_modules").unwrap(),
                Regex::new(r"target").unwrap(),
            ],
            include_patterns: vec![],
        }
    }
}

#[derive(Debug)]
pub struct IndexStats {
    pub files_processed: usize,
    pub directories_processed: usize,
    pub bytes_processed: u64,
    pub errors_count: usize,
    pub duplicates_found: usize,
    pub processing_time: Duration,
}

#[derive(Debug, Clone)]
/// Settings that alter search behavior such as case sensitivity and regex use.
pub struct QueryOptions {
    pub case_sensitive: bool,
    pub regex: bool,
    pub whole_word: bool,
    pub include_content: bool,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            regex: false,
            whole_word: false,
            include_content: false,
        }
    }
}

#[derive(Debug)]
/// A single result from a query containing the record, matched segments and a score.
pub struct SearchResult {
    pub record: FileRecord,
    pub matches: Vec<Match>,
    pub score: f64,
}

#[derive(Debug)]
/// Location information for a term match within a field.
pub struct Match {
    pub field: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// In-memory index that stores file records and provides search and export capabilities.
pub struct FileIndexer {
    records: Arc<Mutex<Vec<FileRecord>>>,
    checksum_map: Arc<Mutex<HashMap<String, Vec<PathBuf>>>>,
    config: IndexConfig,
    stats: Arc<Mutex<IndexStats>>,
}

impl FileIndexer {
    /// Create a new indexer using the provided configuration.
    pub fn new(config: IndexConfig) -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
            checksum_map: Arc::new(Mutex::new(HashMap::new())),
            config,
            stats: Arc::new(Mutex::new(IndexStats {
                files_processed: 0,
                directories_processed: 0,
                bytes_processed: 0,
                errors_count: 0,
                duplicates_found: 0,
                processing_time: Duration::new(0, 0),
            })),
        }
    }
    /// Walk the given directory and populate the in-memory index.
    pub fn index_directory<P: AsRef<Path>>(
        &self,
        root_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        let root_path = root_path.as_ref();
        
        println!("Starting indexing of: {}", root_path.display());
        
        // Collect all file paths first
        let mut file_paths = Vec::new();
        self.collect_file_paths(root_path, 0, &mut file_paths)?;
        
        println!("Found {} files to process", file_paths.len());
        
        // Process files in parallel using rayon
        let records = Arc::clone(&self.records);
        let checksum_map = Arc::clone(&self.checksum_map);
        let stats = Arc::clone(&self.stats);
        let config = self.config.clone();
        
        file_paths.par_iter().for_each(|path| {
            if let Err(e) = Self::process_file(path, &records, &checksum_map, &stats, &config) {
                eprintln!("Error processing {}: {}", path.display(), e);
                stats.lock().unwrap().errors_count += 1;
            }
        });
        
        let elapsed = start_time.elapsed();
        self.stats.lock().unwrap().processing_time = elapsed;
        
        self.detect_duplicates();
        
        println!("Indexing completed in {:.2}s", elapsed.as_secs_f64());
        self.print_stats();
        
        Ok(())
    }
    
    fn collect_file_paths(
        &self,
        dir: &Path,
        depth: usize,
        file_paths: &mut Vec<PathBuf>,
    ) -> io::Result<()> {
        if let Some(max_depth) = self.config.max_depth {
            if depth > max_depth {
                return Ok(());
            }
        }
        
        let entries = fs::read_dir(dir)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if self.should_ignore(&path) {
                continue;
            }
            
            if path.is_dir() {
                self.stats.lock().unwrap().directories_processed += 1;
                
                if self.config.follow_symlinks || !path.is_symlink() {
                    self.collect_file_paths(&path, depth + 1, file_paths)?;
                }
            } else {
                file_paths.push(path);
            }
        }
        
        Ok(())
    }
    
    fn should_ignore(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        
        // Check ignore patterns
        for pattern in &self.config.ignore_patterns {
            if pattern.is_match(&path_str) {
                return true;
            }
        }
        
        // Check include patterns (if any)
        if !self.config.include_patterns.is_empty() {
            for pattern in &self.config.include_patterns {
                if pattern.is_match(&path_str) {
                    return false;
                }
            }
            return true;
        }
        
        false
    }
    
    fn process_file(
        path: &Path,
        records: &Arc<Mutex<Vec<FileRecord>>>,
        checksum_map: &Arc<Mutex<HashMap<String, Vec<PathBuf>>>>,
        stats: &Arc<Mutex<IndexStats>>,
        config: &IndexConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let metadata = fs::metadata(path)?;
        let mut record = Self::create_file_record(path, &metadata)?;
        
        // Compute checksum if requested
        if config.compute_checksums && !metadata.is_dir() {
            if let Ok(checksum) = Self::compute_checksum(path) {
                record.checksum = Some(checksum.clone());
                
                let mut map = checksum_map.lock().unwrap();
                map.entry(checksum).or_insert_with(Vec::new).push(path.to_path_buf());
            }
        }
        
        // Extract content preview if requested
        if config.include_content && !metadata.is_dir() {
            if let Ok(content) = Self::extract_content_preview(path, config.content_preview_size) {
                record.content_preview = Some(content);
            }
        }
        
        records.lock().unwrap().push(record);
        
        let mut stats = stats.lock().unwrap();
        stats.files_processed += 1;
        stats.bytes_processed += metadata.len();
        
        if stats.files_processed % 1000 == 0 {
            println!("Processed {} files...", stats.files_processed);
        }
        
        Ok(())
    }
    
    fn create_file_record(
        path: &Path,
        metadata: &Metadata,
    ) -> Result<FileRecord, Box<dyn std::error::Error>> {
        let name = path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        
        let extension = path.extension()
            .map(|ext| ext.to_string_lossy().to_string());
        
        let permissions = Self::format_permissions(metadata);
        
        let created = metadata.created().ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        
        let modified = metadata.modified()
            .map_err(|_| "Failed to get modified time")
            .and_then(|t| t.duration_since(UNIX_EPOCH).map_err(|_| "Invalid time"))
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        let accessed = metadata.accessed().ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs());
        
        Ok(FileRecord {
            path: path.to_path_buf(),
            name,
            size: metadata.len(),
            created,
            modified,
            accessed,
            permissions,
            is_dir: metadata.is_dir(),
            extension,
            checksum: None,
            content_preview: None,
        })
    }
    
    fn format_permissions(metadata: &Metadata) -> String {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = metadata.permissions().mode();
            format!("{:o}", mode & 0o777)
        }
        
        #[cfg(not(unix))]
        {
            if metadata.permissions().readonly() {
                "r--".to_string()
            } else {
                "rw-".to_string()
            }
        }
    }
    
    fn compute_checksum(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0; 8192];
        
        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    fn extract_content_preview(
        path: &Path,
        max_size: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut buffer = vec![0; max_size];
        
        let bytes_read = reader.read(&mut buffer)?;
        buffer.truncate(bytes_read);
        
        // Try to convert to UTF-8, fallback to lossy conversion
        match String::from_utf8(buffer) {
            Ok(content) => Ok(content),
            Err(e) => Ok(String::from_utf8_lossy(e.as_bytes()).to_string()),
        }
    }
    
    fn detect_duplicates(&self) {
        let checksum_map = self.checksum_map.lock().unwrap();
        let mut duplicates_count = 0;
        
        for (checksum, paths) in checksum_map.iter() {
            if paths.len() > 1 {
                duplicates_count += paths.len() - 1;
                println!("Duplicate files (checksum: {}):", &checksum[..8]);
                for path in paths {
                    println!("  {}", path.display());
                }
            }
        }
        
        self.stats.lock().unwrap().duplicates_found = duplicates_count;
    }
    
    /// Execute a search query against the index using the given options.
    pub fn search(&self, query: &str, options: &QueryOptions) -> Vec<SearchResult> {
        let records = self.records.lock().unwrap();
        let query_parser = QueryParser::new(query, options);
        
        records.par_iter()
            .filter_map(|record| {
                if let Some(matches) = query_parser.matches(record) {
                    let score = Self::calculate_score(&matches, record);
                    Some(SearchResult {
                        record: record.clone(),
                        matches,
                        score,
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }
    
    fn calculate_score(matches: &[Match], record: &FileRecord) -> f64 {
        let mut score = 0.0;
        
        for m in matches {
            match m.field.as_str() {
                "name" => score += 10.0,
                "path" => score += 5.0,
                "content" => score += 2.0,
                _ => score += 1.0,
            }
        }
        
        // Boost score for exact name matches
        if matches.iter().any(|m| m.field == "name" && m.text.len() == record.name.len()) {
            score *= 2.0;
        }
        
        score
    }
    
    /// Write the indexed records to a pretty printed JSON file.
    pub fn export_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let records = self.records.lock().unwrap();
        let file = File::create(path)?;
        serde_json::to_writer_pretty(file, &*records)?;
        Ok(())
    }
    
    /// Export the index in a simple CSV format.
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let records = self.records.lock().unwrap();
        let mut file = File::create(path)?;
        
        writeln!(file, "path,name,size,modified,is_dir,extension,permissions,checksum")?;
        
        for record in records.iter() {
            writeln!(
                file,
                "{},{},{},{},{},{},{},{}",
                record.path.display(),
                record.name,
                record.size,
                record.modified,
                record.is_dir,
                record.extension.as_deref().unwrap_or(""),
                record.permissions,
                record.checksum.as_deref().unwrap_or("")
            )?;
        }
        
        Ok(())
    }
    
    /// Display summary statistics for the most recent indexing run.
    pub fn print_stats(&self) {
        let stats = self.stats.lock().unwrap();
        println!("\nIndexing Statistics:");
        println!("  Files processed: {}", stats.files_processed);
        println!("  Directories processed: {}", stats.directories_processed);
        println!("  Bytes processed: {:.2} MB", stats.bytes_processed as f64 / 1024.0 / 1024.0);
        println!("  Errors encountered: {}", stats.errors_count);
        println!("  Duplicates found: {}", stats.duplicates_found);
        println!("  Processing time: {:.2}s", stats.processing_time.as_secs_f64());
        println!("  Files per second: {:.0}", 
                 stats.files_processed as f64 / stats.processing_time.as_secs_f64());
    }
    
    /// Return groups of files that share the same checksum.
    pub fn get_duplicate_files(&self) -> Vec<(String, Vec<PathBuf>)> {
        let checksum_map = self.checksum_map.lock().unwrap();
        checksum_map.iter()
            .filter(|(_, paths)| paths.len() > 1)
            .map(|(checksum, paths)| (checksum.clone(), paths.clone()))
            .collect()
    }
}

struct QueryParser {
    tokens: Vec<QueryToken>,
    options: QueryOptions,
}

#[derive(Debug, Clone)]
enum QueryToken {
    Term(String),
    And,
    Or,
    Not,
    LeftParen,
    RightParen,
    Wildcard(String),
}

impl QueryParser {
    fn new(query: &str, options: &QueryOptions) -> Self {
        let tokens = Self::tokenize(query);
        Self {
            tokens,
            options: options.clone(),
        }
    }
    
    fn tokenize(query: &str) -> Vec<QueryToken> {
        let mut tokens = Vec::new();
        let mut current_term = String::new();
        let mut chars = query.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                ' ' | '\t' | '\n' => {
                    if !current_term.is_empty() {
                        tokens.push(Self::create_term_token(current_term.clone()));
                        current_term.clear();
                    }
                }
                '(' => {
                    if !current_term.is_empty() {
                        tokens.push(Self::create_term_token(current_term.clone()));
                        current_term.clear();
                    }
                    tokens.push(QueryToken::LeftParen);
                }
                ')' => {
                    if !current_term.is_empty() {
                        tokens.push(Self::create_term_token(current_term.clone()));
                        current_term.clear();
                    }
                    tokens.push(QueryToken::RightParen);
                }
                '"' => {
                    // Handle quoted strings
                    current_term.clear();
                    while let Some(ch) = chars.next() {
                        if ch == '"' {
                            break;
                        }
                        current_term.push(ch);
                    }
                    if !current_term.is_empty() {
                        tokens.push(QueryToken::Term(current_term.clone()));
                        current_term.clear();
                    }
                }
                _ => {
                    current_term.push(ch);
                }
            }
        }
        
        if !current_term.is_empty() {
            tokens.push(Self::create_term_token(current_term));
        }
        
        tokens
    }
    
    fn create_term_token(term: String) -> QueryToken {
        match term.to_uppercase().as_str() {
            "AND" => QueryToken::And,
            "OR" => QueryToken::Or,
            "NOT" => QueryToken::Not,
            _ => {
                if term.contains('*') || term.contains('?') {
                    QueryToken::Wildcard(term)
                } else {
                    QueryToken::Term(term)
                }
            }
        }
    }
    
    fn matches(&self, record: &FileRecord) -> Option<Vec<Match>> {
        let mut all_matches = Vec::new();
        
        // Search in different fields
        self.search_field("name", &record.name, &mut all_matches);
        self.search_field("path", &record.path.to_string_lossy(), &mut all_matches);
        
        if let Some(ref content) = record.content_preview {
            self.search_field("content", content, &mut all_matches);
        }
        
        if !all_matches.is_empty() {
            Some(all_matches)
        } else {
            None
        }
    }
    
    fn search_field(&self, field_name: &str, field_value: &str, matches: &mut Vec<Match>) {
        for token in &self.tokens {
            match token {
                QueryToken::Term(term) => {
                    if let Some(found_matches) = self.find_term_matches(term, field_value) {
                        for (start, end) in found_matches {
                            matches.push(Match {
                                field: field_name.to_string(),
                                text: field_value[start..end].to_string(),
                                start,
                                end,
                            });
                        }
                    }
                }
                QueryToken::Wildcard(pattern) => {
                    if let Some(found_matches) = self.find_wildcard_matches(pattern, field_value) {
                        for (start, end) in found_matches {
                            matches.push(Match {
                                field: field_name.to_string(),
                                text: field_value[start..end].to_string(),
                                start,
                                end,
                            });
                        }
                    }
                }
                _ => {} // Handle boolean operators in more sophisticated implementation
            }
        }
    }
    
    fn find_term_matches(&self, term: &str, text: &str) -> Option<Vec<(usize, usize)>> {
        let mut matches = Vec::new();
        
        let search_text = if self.options.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };
        let search_term = if self.options.case_sensitive {
            term.to_string()
        } else {
            term.to_lowercase()
        };
        
        if self.options.regex {
            if let Ok(regex) = Regex::new(&search_term) {
                for mat in regex.find_iter(&search_text) {
                    matches.push((mat.start(), mat.end()));
                }
            }
        } else if self.options.whole_word {
            let word_regex = Regex::new(&format!(r"\b{}\b", regex::escape(&search_term))).unwrap();
            for mat in word_regex.find_iter(&search_text) {
                matches.push((mat.start(), mat.end()));
            }
        } else {
            let mut start = 0;
            while let Some(pos) = search_text[start..].find(&search_term) {
                let actual_pos = start + pos;
                matches.push((actual_pos, actual_pos + search_term.len()));
                start = actual_pos + 1;
            }
        }
        
        if matches.is_empty() {
            None
        } else {
            Some(matches)
        }
    }
    
    fn find_wildcard_matches(&self, pattern: &str, text: &str) -> Option<Vec<(usize, usize)>> {
        let regex_pattern = pattern
            .replace('*', ".*")
            .replace('?', ".");
        
        if let Ok(regex) = Regex::new(&regex_pattern) {
            let search_text = if self.options.case_sensitive { text } else { &text.to_lowercase() };
            let matches: Vec<_> = regex.find_iter(search_text)
                .map(|mat| (mat.start(), mat.end()))
                .collect();
            
            if matches.is_empty() {
                None
            } else {
                Some(matches)
            }
        } else {
            None
        }
    }
}

fn highlight_match(text: &str, start: usize, end: usize) -> String {
    format!("{}**{}**{}", 
            &text[..start], 
            &text[start..end], 
            &text[end..])
}

// Example usage and CLI interface
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: {} <command> [options]", args[0]);
        println!("Commands:");
        println!("  index <directory> [--depth N] [--content] [--checksums]");
        println!("  search <query> [--regex] [--case-sensitive]");
        println!("  export <format> <output_file>");
        println!("  duplicates");
        return Ok(());
    }
    
    match args[1].as_str() {
        "index" => {
            if args.len() < 3 {
                println!("Usage: {} index <directory> [options]", args[0]);
                return Ok(());
            }
            
            let mut config = IndexConfig::default();
            
            // Parse additional arguments
            for i in 3..args.len() {
                match args[i].as_str() {
                    "--content" => config.include_content = true,
                    "--checksums" => config.compute_checksums = true,
                    "--depth" => {
                        if i + 1 < args.len() {
                            config.max_depth = args[i + 1].parse().ok();
                        }
                    }
                    _ => {}
                }
            }
            
            let indexer = FileIndexer::new(config);
            indexer.index_directory(&args[2])?;
            
            // Save index for later use
            indexer.export_json("index.json")?;
            println!("Index saved to index.json");
        }
        
        "search" => {
            if args.len() < 3 {
                println!("Usage: {} search <query> [options]", args[0]);
                return Ok(());
            }
            
            // Load existing index
            let indexer = FileIndexer::new(IndexConfig::default());
            // In a real implementation, you'd load from saved index
            
            let mut options = QueryOptions::default();
            for arg in &args[3..] {
                match arg.as_str() {
                    "--regex" => options.regex = true,
                    "--case-sensitive" => options.case_sensitive = true,
                    "--content" => options.include_content = true,
                    _ => {}
                }
            }
            
            let results = indexer.search(&args[2], &options);
            
            println!("Found {} results:", results.len());
            for (i, result) in results.iter().take(20).enumerate() {
                println!("{}. {} (score: {:.2})", 
                         i + 1, 
                         result.record.path.display(), 
                         result.score);
                for m in &result.matches {
                    println!("   {}: {}", m.field, highlight_match(&m.text, m.start, m.end));
                }
            }
        }
        
        "duplicates" => {
            let indexer = FileIndexer::new(IndexConfig::default());
            let duplicates = indexer.get_duplicate_files();
            
            if duplicates.is_empty() {
                println!("No duplicate files found.");
            } else {
                println!("Found {} groups of duplicate files:", duplicates.len());
                for (checksum, paths) in duplicates {
                    println!("\nChecksum: {}", &checksum[..16]);
                    for path in paths {
                        println!("  {}", path.display());
                    }
                }
            }
        }
        
        _ => {
            println!("Unknown command: {}", args[1]);
        }
    }
    
    Ok(())
}

// Add required dependencies to Cargo.toml:
/*
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
regex = "1.11"
rayon = "1.10"
num_cpus = "1.17"
*/