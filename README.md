# 🔮 ARCANE: Academic Research & Citation Analysis Network Engine

ARCANE is a Model Context Protocol (MCP) server that provides unified access to academic paper databases including arXiv, Semantic Scholar, and OpenCitations. Features intelligent identifier resolution, citation analysis, and citation graph generation.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- macOS, Linux, or Windows
- Claude Desktop installed

### 2. Installation

```bash
# Clone or create the project directory
git clone <repository-url>
cd ARCANE

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 3. Configure Claude Desktop

#### Find Your Claude Desktop Config File

**macOS:**
```bash
~/Library/Application Support/Claude/claude_desktop_config.json
```

**Windows:**
```
%APPDATA%\Claude\claude_desktop_config.json
```

**Linux:**
```bash
~/.config/Claude/claude_desktop_config.json
```

#### Edit the Configuration

Add the MCP server to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "academic-discovery": {
      "command": "/path/to/your/ARCANE/arcane-wrapper.sh",
      "env": {
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

**Important:** Replace `/path/to/your/ARCANE/` with the actual path to your project directory.

### 4. Restart Claude Desktop

After updating the configuration file, restart Claude Desktop for the changes to take effect.

### 5. Test It!

Once Claude Desktop restarts, you can test the MCP server by asking:

- "Search for papers about transformer neural networks"
- "Get details for arXiv paper 1706.03762" (iykyk)
- "Find all identifiers for that paper"
- "Get citation data for DOI 10.1038/nature14539"
- "Build a citation graph for that paper"

## ✨ Features

- 🔍 **Unified Search**: Search across arXiv and Semantic Scholar simultaneously
- 🔗 **Smart ID Resolution**: Automatically maps DOI ↔ arXiv ↔ Semantic Scholar IDs
- 📊 **Citation Analysis**: Get citation data from OpenCitations
- 🕸️ **Citation Graphs**: Build network visualizations of paper relationships
- 💾 **Intelligent Caching**: SQLite database prevents duplicate API calls

## 🛠️ Available Tools

| Tool | Description | Best Identifier |
|------|-------------|----------------|
| `search_papers` | Multi-source paper search | Any (natural language queries) |
| `get_paper` | Detailed paper information | arXiv ID, DOI, or Semantic Scholar ID |
| `resolve_identifiers` | Cross-platform ID mapping | Any identifier type |
| `get_citations` | Citation and reference data | **DOI recommended** |
| `build_citation_graph` | Network visualization data | **DOI required** |

## 📋 Identifier Compatibility

ARCANE supports multiple paper identifiers, but not all tools work equally well with all identifier types:

- **arXiv IDs** (e.g., `2006.12469`): Best for paper details and metadata
- **DOIs** (e.g., `10.1088/2632-2153/ac362b`): Best for citations and impact analysis  
- **Semantic Scholar IDs**: Good for cross-referencing

**For citation analysis and graphs, DOI identifiers are strongly recommended.**

See `IDENTIFIER_COMPATIBILITY.md` for detailed compatibility information.

## 🔧 Configuration

### Environment Variables

Create a `.env` file in your project root for API keys and configuration:

```bash
# API Keys (optional but recommended)
SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
OPENCITATIONS_ACCESS_TOKEN=your_token_here

# Database
ARCANE_DB_PATH=./data/academic_papers.db

# Logging
LOG_LEVEL=INFO

# Rate Limits
ARXIV_RATE_LIMIT=1.0
S2_RATE_LIMIT=1.0
OC_RATE_LIMIT=2.0
```

### Getting API Keys

**Semantic Scholar API Key:**
1. Visit https://www.semanticscholar.org/product/api
2. Sign up for a free API key
3. Add to your `.env` file

**OpenCitations Access Token:**
1. Visit https://opencitations.net/
2. Register for an account
3. Generate an access token
4. Add to your `.env` file

## 🎯 The Identifier Resolution Problem

The core innovation of this MCP server is solving cross-platform paper identification. For example:

- **arXiv ID**: `1706.03762` 
- **DOI**: `10.48550/arXiv.1706.03762`
- **Semantic Scholar ID**: `13756489`

All refer to the same paper ("Attention Is All You Need"), but each platform uses different identifiers. The resolver creates unified mappings with confidence scoring.

**Next Steps**: 
1. Add your API keys to get real data
2. The identifier resolution and caching systems are production-ready
3. All MCP tools work end-to-end

## 📈 Architecture Highlights

- **Rate Limiting**: Respects each API's constraints automatically
- **Async Design**: Non-blocking I/O for all operations  
- **Error Handling**: Graceful degradation when APIs are unavailable
- **Extensible**: Easy to add new data sources
- **Testable**: Modular design with clear interfaces

## 🔍 Troubleshooting

### Common Issues

**1. "Import could not be resolved" errors**
- Make sure you're using the virtual environment: `source venv/bin/activate`
- Reinstall the package: `pip install -e .`

**2. Claude Desktop can't find the server**
- Check that the path in your config file is correct
- Ensure the wrapper script is executable: `chmod +x arcane-wrapper.sh`
- Verify the virtual environment exists and is properly set up

**3. Server starts but Claude Desktop doesn't connect**
- Check the Claude Desktop logs for connection errors
- Ensure the server is running: `./arcane-wrapper.sh`
- Restart Claude Desktop after configuration changes

**4. Permission denied errors**
- Make sure the wrapper script is executable: `chmod +x arcane-wrapper.sh`
- Check file permissions on the project directory

### Manual Testing

Test the server manually:
```bash
# Activate virtual environment
source venv/bin/activate

# Test the server directly
python src/arcane_mcp/server.py

# Or use the wrapper script
./arcane-wrapper.sh
```

You should see: `INFO:arcane: MCP Server initialized` in the terminal.

## 🤝 Contributing

This is a complete, working implementation ready for production use. Areas for enhancement:

- Additional data sources (PubMed, DBLP, etc.)
- Enhanced fuzzy matching algorithms
- Advanced citation analysis features
- Performance optimizations

## 📝 License

Apache 2.0 License - see LICENSE for details.

---

**Built with ❤️ for the academic research community**
