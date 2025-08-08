#!/usr/bin/env python3
"""
SearchYourCodes - Intelligent Code Search and Discovery Platform

A Flask web application that provides an intelligent code search interface
using multiple search methods: exact keyword matching, code structure analysis
with UniXcoder, and semantic search with SBERT.
"""

import os
import sys
import signal
import atexit
import gc
from typing import Dict, List, Any, Optional, Tuple
from flask import Flask, render_template, request, jsonify

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from core.search import compare_models
except ImportError as e:
    print(f"Error importing search modules: {e}")
    print("Make sure the core modules are available")
    sys.exit(1)

app = Flask(__name__, template_folder='templates')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size (desktop can handle more)
DEFAULT_MAX_RESULTS = 10  # More results for desktop users
MAX_CODE_DISPLAY_LINES = 50  # More lines for desktop viewing
MAX_CODE_HEIGHT = 600  # Larger height for desktop screens


def cleanup_resources():
    """Clean up resources on shutdown"""
    print("\n🧹 Cleaning up resources...")
    # Force garbage collection to help clean up multiprocessing resources
    gc.collect()


def signal_handler(sig, frame):
    """Handle SIGINT (Ctrl+C) gracefully"""
    print("\n🛑 Shutting down SearchYourCodes gracefully...")
    cleanup_resources()
    sys.exit(0)


# Register cleanup handlers
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_resources)


@app.route('/')
def index():
    """Main search interface"""
    return render_template('index.html')


@app.route('/test')
def test():
    """Test endpoint to verify template updates"""
    return "<h1>Template Test - SearchYourCodes is working!</h1><p>Layout: Responsive 3-Column Grid</p>"


@app.route('/open-file')
def open_file():
    """Display file content in browser with line highlighting"""
    file_path = request.args.get('file', '').strip()
    line = request.args.get('line', '1')
    line_end = request.args.get('line_end', line)
    
    if not file_path:
        return "No file specified", 400
    
    # Sanitize and resolve file path
    try:
        file_path = _resolve_file_path(file_path)
        start_line, end_line = _parse_line_numbers(line, line_end)
        
        # Security check: ensure file is within project directory
        project_root = os.path.dirname(os.path.dirname(__file__))  # Go up one level from app/
        if not os.path.commonpath([file_path, project_root]) == project_root:
            return "Access denied: File outside project directory", 403
            
        if not os.path.exists(file_path):
            return f"File not found: {file_path}", 404
            
        return _generate_file_viewer_html(file_path, start_line, end_line)
        
    except ValueError as e:
        return f"Invalid parameters: {str(e)}", 400
    except Exception as e:
        return f"Error reading file: {str(e)}", 500


@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        query = data.get('query', '').strip()
        max_results = data.get('max_results', DEFAULT_MAX_RESULTS)
        
        # Validate input
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
            
        if not isinstance(max_results, int) or max_results < 1 or max_results > 100:  # Allow more results for desktop
            max_results = DEFAULT_MAX_RESULTS
        
        # Perform all three searches
        keyword_results, unixcoder_results, sbert_results = compare_models(query, k=max_results)
        
        # Format results for web display
        results = {
            'query': query,
            'exact_match': _format_keyword_results(keyword_results),
            'code_structure': _format_semantic_results(unixcoder_results, 'UniXcoder'),
            'semantic': _format_semantic_results(sbert_results, 'SBERT')
        }
        
        return jsonify(results)
    
    except Exception as e:
        app.logger.error(f"Search failed: {str(e)}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


@app.route('/debug-search')
def debug_search():
    """Debug endpoint to check search results format"""
    try:
        query = request.args.get('q', 'test')
        keyword_results, unixcoder_results, sbert_results = compare_models(query, k=2)
        
        debug_info = {
            'query': query,
            'keyword_sample': keyword_results[:1] if keyword_results else [],
            'unixcoder_sample': unixcoder_results[:1] if unixcoder_results else [],
            'sbert_sample': sbert_results[:1] if sbert_results else []
        }
        
        return f"<pre>{str(debug_info)}</pre>"
    except Exception as e:
        return f"Error: {str(e)}"


# Helper functions
def _resolve_file_path(file_path: str) -> str:
    """Resolve and sanitize file path using configuration"""
    if not file_path:
        raise ValueError("Empty file path")
    
    # If already absolute path, return as is
    if os.path.isabs(file_path):
        return file_path
    
    # Get project root (parent of app directory)
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Try to get codebase path from config
    try:
        sys.path.insert(0, os.path.join(project_root, 'core'))
        from config import resolve_codebase_path
        codebase_path = resolve_codebase_path()
        codebase_relative = os.path.relpath(codebase_path, project_root)
    except:
        # Fallback to default path
        codebase_relative = 'data/codebases/manuel_natcom/src'
    
    # Try different possible locations for the file
    possible_paths = [
        # Try as direct relative path from project root
        os.path.join(project_root, file_path),
        # Try in the configured codebase directory
        os.path.join(project_root, codebase_relative, file_path),
        # Try one level up from configured codebase
        os.path.join(os.path.dirname(os.path.join(project_root, codebase_relative)), file_path),
        # Try in src directory (fallback)
        os.path.join(project_root, 'src', file_path),
    ]
    
    # Return the first path that exists
    for path in possible_paths:
        if os.path.exists(path):
            return os.path.abspath(path)
    
    # If none exist, return the most likely path (configured codebase location)
    return os.path.abspath(os.path.join(project_root, codebase_relative, file_path))


def _parse_line_numbers(line: str, line_end: str) -> Tuple[int, int]:
    """Parse and validate line numbers"""
    try:
        start_line = int(line) if line else 1
        end_line = int(line_end) if line_end else start_line
    except (ValueError, TypeError):
        start_line = 1
        end_line = 1
    
    # Ensure valid range
    start_line = max(1, start_line)
    end_line = max(start_line, end_line)
    
    return start_line, end_line


def _generate_file_viewer_html(file_path: str, start_line: int, end_line: int) -> str:
    """Generate HTML for file viewer"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
    except Exception as e:
        raise Exception(f"Could not read file: {str(e)}")
    
    lines = file_content.split('\n')
    
    # Generate line-numbered content with highlighting
    numbered_lines = []
    for i, line_content in enumerate(lines, 1):
        line_class = 'highlight-line' if start_line <= i <= end_line else ''
        numbered_lines.append({
            'number': i,
            'content': line_content,
            'class': line_class
        })
    
    # Create line info text
    if start_line == end_line:
        line_info = f"📍 Highlighting line {start_line}"
    else:
        line_info = f"📍 Highlighting lines {start_line}-{end_line}"
    
    # Generate HTML with embedded CSS
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{os.path.basename(file_path)} - Code Viewer</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
                margin: 0;
                padding: 20px;
                background: #f8f9fa;
                font-size: 14px;
                line-height: 1.5;
            }}
            .container {{
                max-width: 98%;
                margin: 0 auto;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .header {{
                background: #2c3e50;
                color: white;
                padding: 20px 25px;
                border-radius: 8px 8px 0 0;
            }}
            .file-name {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            .file-path {{
                font-size: 13px;
                color: #bdc3c7;
                word-break: break-all;
            }}
            .line-info {{
                font-size: 13px;
                color: #f39c12;
                margin-top: 8px;
            }}
            .code-container {{
                background: #ffffff;
                border-radius: 0 0 8px 8px;
                overflow: auto;
                max-height: 85vh;
            }}
            .code-line {{
                display: flex;
                border-bottom: 1px solid #f1f2f6;
                min-height: 22px;
            }}
            .code-line:hover {{
                background: #f8f9fa;
            }}
            .line-number {{
                background: #f1f2f6;
                color: #666;
                padding: 4px 12px;
                text-align: right;
                min-width: 60px;
                border-right: 1px solid #e1e5e9;
                user-select: none;
                font-size: 12px;
            }}
            .line-content {{
                padding: 4px 16px;
                white-space: pre;
                flex: 1;
                overflow-x: auto;
            }}
            .highlight-line {{
                background: #fff3cd !important;
                border-left: 4px solid #f39c12;
            }}
            .highlight-line .line-number {{
                background: #f39c12;
                color: white;
                font-weight: bold;
            }}
            .back-button {{
                background: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                margin-top: 15px;
                font-size: 14px;
            }}
            .back-button:hover {{
                background: #2980b9;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="file-name">📄 {os.path.basename(file_path)}</div>
                <div class="file-path">{file_path}</div>
                <div class="line-info">{line_info}</div>
                <button class="back-button" onclick="window.close()">← Close</button>
            </div>
            <div class="code-container">
                {''.join(f'''
                <div class="code-line {line_data["class"]}">
                    <div class="line-number">{line_data["number"]}</div>
                    <div class="line-content">{_escape_html(line_data["content"])}</div>
                </div>''' for line_data in numbered_lines)}
            </div>
        </div>
        
        <script>
            // Scroll to the first highlighted line
            const firstHighlightedLine = document.querySelector('.highlight-line');
            if (firstHighlightedLine) {{
                firstHighlightedLine.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        </script>
    </body>
    </html>
    """


def _escape_html(text: str) -> str:
    """Escape HTML characters in text"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _read_file_lines(file_path: str, start_line: int, end_line: int) -> str:
    """Read specific lines from a file"""
    try:
        # Handle relative paths - resolve using the same logic as _resolve_file_path
        if not os.path.isabs(file_path):
            file_path = _resolve_file_path(file_path)
        
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"
        
        # Ensure line numbers are valid
        start_line = max(1, int(start_line)) if start_line is not None else 1
        end_line = max(start_line, int(end_line)) if end_line is not None else start_line
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Convert to 0-based indexing
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        if start_idx >= len(lines):
            return "Line number out of range"
        
        # Extract the requested lines
        selected_lines = lines[start_idx:end_idx]
        
        # Format with line numbers
        formatted_lines = []
        for i, line in enumerate(selected_lines):
            line_num = start_line + i
            formatted_lines.append(f"{line_num:3}: {line.rstrip()}")
        
        return '\\n'.join(formatted_lines)
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


def _format_keyword_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format keyword search results"""
    if not results:
        return []
    
    formatted = []
    for result in results:
        try:
            metadata = result.get('metadata', {})
            
            # Extract function name from metadata
            function_name = (metadata.get('function_name') or 
                           metadata.get('class_name') or
                           'Code Match')
            
            # Extract file path from metadata
            file_path = metadata.get('file_path', 'Unknown')
            
            # Extract line numbers from metadata
            line_start = metadata.get('start_line', 1)
            line_end = metadata.get('end_line', line_start)
            
            # Get match type for additional info
            match_type = metadata.get('match_type', 'content')
            
            # Read the actual code content from the file at the specific lines
            code_content = _read_file_lines(file_path, line_start, line_end)
            
            formatted.append({
                'function': function_name,
                'file': file_path,
                'code': code_content,
                'has_score': False,
                'line_start': line_start,
                'line_end': line_end,
                'match_type': match_type
            })
        except Exception as e:
            app.logger.warning(f"Error formatting keyword result: {e}")
            continue
    
    return formatted


def _format_semantic_results(results: List[Dict[str, Any]], method_name: str) -> List[Dict[str, Any]]:
    """Format semantic search results (UniXcoder/SBERT)"""
    if not results:
        return []
    
    formatted = []
    for i, result in enumerate(results):
        try:
            metadata = result.get('metadata', {})
            distance = result.get('distance', 0)
            
            # Convert distance to similarity score (ChromaDB uses cosine distance)
            if distance <= 1:
                similarity = 1 - distance
            else:
                similarity = 1 / (1 + distance)
            
            # Ensure similarity is between 0 and 1
            similarity = max(0, min(1, similarity))
            
            # Get file path and line numbers from metadata
            file_path = metadata.get('file_path', 'Unknown')
            line_start = metadata.get('line_start', metadata.get('start_line', 1))
            line_end = metadata.get('line_end', metadata.get('end_line', line_start))
            
            # Read the actual code content from the file at the specific lines
            code_content = _read_file_lines(file_path, line_start, line_end)
            
            formatted.append({
                'function': metadata.get('function_name', f'Result {i+1}'),
                'file': file_path,
                'code': code_content,
                'score': round(similarity, 3),
                'has_score': True,
                'line_start': line_start,
                'line_end': line_end
            })
        except Exception as e:
            app.logger.warning(f"Error formatting {method_name} result: {e}")
            continue
    
    return formatted


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🔍 Starting SearchYourCodes - Desktop Code Search Platform...")
    print("🖥️  Optimized for desktop environments")
    print("📋 Features:")
    print("   • Exact Match: Direct keyword search in code")
    print("   • Code Structure: UniXcoder AI semantic matching")
    print("   • Semantic Search: SBERT natural language understanding")
    print("   • Desktop-optimized UI with larger displays and more results")
    print("\n📦 Dependencies:")
    print("   Make sure Flask is installed: pip install flask")
    print("\n🌐 Access the interface at: http://localhost:8081")
    print("🚀 Ready to search your codebase!")
    print("💡 Press Ctrl+C to shutdown gracefully")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=8081)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
