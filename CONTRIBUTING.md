# Contributing to PRSM

Welcome to the PRSM community! PRSM is a P2P infrastructure protocol for open-source collaboration — a mesh network where consumer-class nodes share latent storage, compute, and data, and any third-party LLM can reach the network through MCP (Model Context Protocol) tools. Contributors earn FTNS tokens for running nodes and publishing data.

## 🎯 Our Mission

Build an open-source commons for AI infrastructure. Aggregate the latent resources of consumer electronics into a P2P mesh. Let any LLM — local, OAuth, or API — use that mesh via MCP tools. Pay contributors in FTNS so sharing beats hoarding.

**What PRSM is not:** It's not an AGI framework, not a teacher/distillation platform, not a hosted-model marketplace. Reasoning happens in the caller's LLM. PRSM supplies infrastructure (WASM sandboxes, semantic shards, TEE-backed compute, FTNS settlement).

## 🚀 Getting Started

### **Prerequisites**

- **Python 3.9+** (3.11+ recommended for best performance)
- **Git** for version control
- **Basic understanding** of AI/ML concepts
- **Enthusiasm** for collaborative science!

### **Setting Up Your Development Environment**

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork locally
git clone https://github.com/YOUR-USERNAME/PRSM.git
cd PRSM

# 3. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 5. Set up environment configuration
cp .env.example .env
# Edit .env file with your own API keys and configuration

# 6. Run tests to ensure everything works
pytest --timeout=120

# 7. Run the smoke test to verify core functionality
python scripts/smoke_test.py

# 8. Create a new branch for your contribution
git checkout -b feature/your-feature-name
```

### **Project Structure Overview**

```
PRSM/
├── prsm/                           # Main source code
│   ├── compute/
│   │   ├── wasm/                   # Ring 1: Wasmtime sandbox + hardware profiler
│   │   ├── agents/                 # Ring 2 + Ring 5: mobile agent dispatch + WASM agent runtime
│   │   ├── swarm/                  # Ring 3: semantic sharding + swarm coordinator
│   │   ├── tee/                    # Ring 7: TEE runtime abstraction + DP noise
│   │   ├── model_sharding/         # Ring 8: tensor-parallel model sharding
│   │   ├── nwtn/training/          # Ring 9: NWTN training pipeline (AgentTrace → JSONL)
│   │   ├── security/               # Ring 10: integrity, privacy budget, audit chain
│   │   ├── federation/             # P2P node infrastructure
│   │   └── chronos/                # FTNS ↔ USD/USDT bridge
│   ├── economy/
│   │   └── tokenomics/             # Ring 4: FTNS ledger + pricing + staking
│   ├── storage/                    # Native ContentStore (post-v1.5.0)
│   ├── node/                       # Node implementation + content economy
│   ├── interface/api/              # HTTP API + middleware
│   ├── cli.py                      # Main CLI
│   ├── cli_modules/                # CLI subcommand modules
│   ├── mcp_server.py               # MCP server exposing 16 PRSM tools
│   └── core/                       # Models, config, DB, crypto
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   ├── security/                   # Security tests
│   └── performance/                # Performance tests
├── docs/                           # Documentation
│   ├── api/                        # API documentation
│   ├── performance/                # Performance documentation
│   ├── tokenomics/                 # Tokenomics documentation
│   └── metadata/                   # Architecture metadata
├── scripts/                        # Utility scripts
│   ├── smoke_test.py               # End-to-end smoke test
│   └── release/                    # Release management
├── sdks/                           # SDK packages
├── config/                         # Configuration files
├── docker/                         # Docker configurations
│   ├── monitoring/                 # Monitoring configurations
│   └── Dockerfile.*                # Docker images
├── PRSM_ui_mockup/                 # Web interface and real-time UI
│   ├── js/                         # JavaScript API client and WebSocket integration
│   ├── css/                        # Responsive styling and themes
│   └── test_*.html                 # Testing interfaces for UI validation
├── examples_and_demos/             # Usage examples and demos
├── tools/                          # Development tools
├── utilities/                      # Utility scripts
├── CONTRIBUTING.md                 # Contribution guidelines
├── README.md                       # Main project documentation
└── pytest.ini                      # Test configuration
```

## 🤝 Ways to Contribute

### **1. 🐛 Bug Reports & Issues**

Help us identify and fix issues:

- **Search existing issues** before creating new ones
- **Use our issue templates** for bug reports and feature requests
- **Provide detailed information**: OS, Python version, error messages, steps to reproduce
- **Include test cases** when possible

**Bug Report Template:**
```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Import '...'
2. Call method '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.11.2]
- PRSM version: [e.g. 1.0.0-beta]

**Additional context**
Any other context about the problem.
```

### **2. 💡 Feature Requests & Ideas**

Propose new capabilities and enhancements:

- **Check the roadmap** first - your idea might already be planned!
- **Describe the use case** - why is this feature needed?
- **Provide examples** of how it would work
- **Consider integration** with existing systems

### **3. 🔧 Code Contributions**

We welcome code contributions of all sizes:

#### **Good First Issues**
Look for issues labeled `good first issue` or `help wanted`:
- Documentation improvements
- Test case additions
- Small bug fixes
- Performance optimizations
- Example implementations

#### **Major Contributions**
For larger contributions:
- **Discuss first** - open an issue to discuss your approach
- **Follow our architecture** - maintain consistency with existing patterns
- **Include tests** - all new code should have comprehensive tests
- **Update documentation** - keep docs in sync with code changes

#### **Code Style Guidelines**

```python
# Use type hints
async def process_query(self, user_input: str, context_allocation: int) -> PRSMResponse:
    """Process user query with context allocation.
    
    Args:
        user_input: The user's research question
        context_allocation: FTNS tokens allocated for processing
        
    Returns:
        Processed response with reasoning trace
    """
    pass

# Use descriptive variable names
response_confidence = 0.95
model_performance_metrics = {"accuracy": 0.88, "latency": 120}

# Follow async/await patterns consistently
async def coordinate_agents(self, clarified_prompt: ClarifiedPrompt) -> AgentPipeline:
    results = await asyncio.gather(
        self.prompt_optimizer.optimize_for_domain(prompt, domain),
        self.model_router.match_to_specialist(task),
        return_exceptions=True
    )
    return AgentPipeline(results=results)
```

#### **Testing Requirements**

All contributions must include appropriate tests:

```python
# Unit tests for individual components
async def test_prompt_optimization():
    optimizer = PromptOptimizer()
    result = await optimizer.optimize_for_domain("test prompt", "science")
    assert result is not None
    assert len(result) > 0

# Integration tests for component interactions
async def test_agent_pipeline():
    # Test the full prompter -> router -> compiler pipeline
    pass

# Performance tests for optimization
async def test_concurrent_processing():
    # Test system under load
    pass
```

### **4. 📖 Documentation Contributions**

Help improve our documentation:

#### **Types of Documentation Needed**
- **API documentation** - Detailed method and class documentation
- **Tutorials** - Step-by-step guides for common tasks
- **Use case examples** - Real-world applications in specific domains
- **Architecture explanations** - Deep dives into system design
- **Troubleshooting guides** - Common issues and solutions

#### **Documentation Standards**
```python
class PromptOptimizer:
    """Advanced prompt optimization for domain-specific AI tasks.
    
    The PromptOptimizer enhances user prompts for better performance
    with specialized AI models across different scientific domains.
    
    Attributes:
        agent_id: Unique identifier for this optimizer instance
        domains: Supported domain categories for optimization
        safety_rules: Content safety guidelines and restrictions
        
    Example:
        >>> optimizer = PromptOptimizer()
        >>> optimized = await optimizer.optimize_for_domain(
        ...     "analyze climate data", "environmental_science"
        ... )
        >>> print(optimized)
        "Perform comprehensive climate data analysis focusing on..."
    """
```

### **5. 🧪 Testing & Validation**

Help ensure PRSM works reliably:

- **Run existing test suites** and report failures
- **Test on different environments** (OS, Python versions)
- **Create test cases** for edge cases and error conditions
- **Performance benchmarking** on different hardware
- **Real-world validation** with actual research use cases

### **6. 🖥️ Web Interface & UI Contributions**

Help improve the PRSM web interface:

#### **Frontend Development**
- **React/JavaScript development** - Enhance real-time UI components
- **WebSocket integration** - Improve streaming and live updates
- **Responsive design** - Mobile and tablet optimization
- **Accessibility** - Screen reader support and keyboard navigation
- **Performance optimization** - Reduce bundle size and improve loading

#### **UX/UI Design**
- **User experience research** - How researchers interact with AI systems
- **Interface design** - Mockups and prototypes for new features
- **Usability testing** - Testing interface flows with real users
- **Design system** - Consistent styling and component library

#### **Testing Web Features**
```bash
# Test the web interface
cd PRSM_ui_mockup
python -m http.server 8080

# Test WebSocket functionality
open http://localhost:8080/test_websocket.html

# Test API integration
open http://localhost:8080/test_integration.html
```

### **7. 🌍 Domain Expertise**

Share your research domain knowledge:

- **Scientific use cases** - How PRSM can help in your field
- **Domain-specific optimizations** - Improvements for specialized applications
- **Validation datasets** - Test data for specific research areas
- **Expert feedback** - Evaluation of PRSM's scientific capabilities

## 🏗️ Development Workflow

### **Standard Contribution Process**

1. **Fork & Clone** - Get your own copy of the repository
2. **Create Branch** - Use descriptive branch names: `feature/agent-optimization` or `fix/consensus-bug`
3. **Develop & Test** - Write code, add tests, ensure everything passes
4. **Documentation** - Update docs to reflect your changes
5. **Commit** - Use clear, descriptive commit messages
6. **Push & PR** - Push to your fork and create a pull request
7. **Review & Iterate** - Respond to feedback and make improvements
8. **Merge** - Once approved, your contribution becomes part of PRSM!

### **Commit Message Guidelines**

```bash
# Good commit messages
git commit -m "feat: add Byzantine consensus validation in P2P network"
git commit -m "fix: resolve memory leak in WASM agent dispatch"
git commit -m "docs: add API documentation for governance system"
git commit -m "test: increase coverage for safety monitoring"

# Prefixes to use:
# feat:     New feature
# fix:      Bug fix
# docs:     Documentation only changes
# test:     Adding missing tests
# refactor: Code change that neither fixes a bug nor adds a feature
# perf:     Performance improvement
# style:    Formatting, missing semicolons, etc.
```

### **Pull Request Guidelines**

**PR Title Format:**
```
[Component] Brief description of changes

Examples:
[Agents] Implement parallel processing in RouterAI
[Safety] Add emergency halt mechanism for critical threats
[Tests] Increase coverage for tokenomics system
```

**PR Description Template:**
```markdown
## Summary
Brief description of what this PR does and why.

## Changes Made
- List specific changes
- Include any new files or major modifications
- Note any breaking changes

## Testing
- [ ] Unit tests pass (`pytest --timeout=120`)
- [ ] Smoke test passes (`python scripts/smoke_test.py`)
- [ ] Integration tests pass
- [ ] Added new tests for new functionality
- [ ] Tested on [specific environment/conditions]

## Documentation
- [ ] Updated relevant documentation
- [ ] Added docstrings for new functions/classes
- [ ] Updated examples if applicable

## Related Issues
Fixes #123
Related to #456

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Changes work as expected
- [ ] No performance regressions introduced
```

### **Pre-PR Checklist**

Before submitting a pull request, ensure all of the following pass:

```bash
# 1. Run the full test suite with timeout
pytest --timeout=120

# 2. Run the smoke test (end-to-end validation)
python scripts/smoke_test.py

# 3. For P2P/federation changes, run the two-node integration test
pytest tests/integration/test_two_node_network.py -v --timeout=120
```

### **Running Tests**

#### **Unit Tests**
```bash
# Run all unit tests
pytest tests/unit/ --timeout=120

# Run specific test file
pytest tests/unit/test_specific_module.py -v

# Run with coverage
pytest --cov=prsm --cov-report=term-missing --cov-report=html
```

#### **Integration Tests**
```bash
# Run all integration tests
pytest tests/integration/ -v --timeout=120

# Run specific integration test
pytest tests/integration/test_two_node_network.py -v
```

#### **Security Tests**
```bash
# Run security test suite
pytest tests/security/ -v --timeout=120
```

#### **Smoke Test**
The smoke test validates the core node workflow end-to-end without external dependencies:
```bash
python scripts/smoke_test.py
```

This test:
- Creates a PRSM node with in-memory configuration
- Initializes and starts the node
- Submits a benchmark compute job
- Verifies results and FTNS balance changes

#### **Local Two-Node P2P Test**

For testing P2P functionality locally, use the two-node integration test:

```bash
# Run the two-node network tests
pytest tests/integration/test_two_node_network.py -v --timeout=120
```

This test validates:
- Node discovery via bootstrap
- Compute job submission and execution between nodes
- Payment settlement between nodes
- Embedding job processing

To run a manual two-node test:
```python
import asyncio
from prsm.node.node import PRSMNode
from prsm.node.config import NodeConfig, NodeRole

async def test_two_nodes():
    # Create bootstrap node
    config_a = NodeConfig(
        display_name="node-a",
        roles=[NodeRole.FULL],
        p2p_port=19400,
        data_dir="/tmp/prsm_node_a",
    )
    node_a = PRSMNode(config_a)
    await node_a.initialize()
    await node_a.start()
    
    # Create second node connecting to bootstrap
    config_b = NodeConfig(
        display_name="node-b",
        roles=[NodeRole.FULL],
        p2p_port=19401,
        data_dir="/tmp/prsm_node_b",
        bootstrap_nodes=["127.0.0.1:19400"],
    )
    node_b = PRSMNode(config_b)
    await node_b.initialize()
    await node_b.start()
    
    # Nodes can now communicate via P2P
    # ... test P2P functionality ...

asyncio.run(test_two_nodes())
```

## 🎯 Current Priority Areas

PRSM is live on Base mainnet, with focus on production-ready P2P
infrastructure. High-priority contribution areas include:

### **1. Performance Optimization**
- **Bottleneck identification** in agent coordination
- **Caching strategies** for improved response times
- **Database query optimization** for large-scale data
- **Memory usage improvements** across all components

### **2. API Standardization**
- **Method signature alignment** across components
- **Error handling consistency** throughout the system
- **Response format standardization** for better integration
- **Parameter validation** and type checking improvements

### **3. P2P Infrastructure**
- **Node hardening** reliability under hostile network conditions
- **Multi-region bootstrap** expansion beyond the current single-region host
- **Storage layer** native ContentStore replication and proofs
- **Economic model validation** FTNS tokenomics and marketplace testing

### **4. Community & Documentation**
- **Getting started tutorials** for new users
- **Research use case examples** in specific domains
- **API reference** completion and improvement
- **Video tutorials** and interactive examples

## 🔬 Research Collaboration Opportunities

### **Academic Partnerships**
- **University collaborations** for research validation
- **Student projects** and thesis opportunities
- **Grant applications** for PRSM-based research
- **Conference presentations** and academic papers

### **Industry Applications**
- **Enterprise R&D** use cases and requirements
- **Commercial deployment** scenarios and optimization
- **Industry-specific** customizations and integrations
- **Business model development** for sustainable adoption

### **Open Source Ecosystem**
- **Integration** with other AI/ML tools and libraries
- **Interoperability** with existing research platforms
- **Community building** through workshops and events
- **Developer tool creation** for easier PRSM adoption

## 🏆 Recognition

We believe in recognizing our contributors:

### **Recognition Programs**
- **Contributor spotlight** in project documentation
- **Conference speaking** opportunities
- **Research collaboration** invitations
- **Advisory board** positions for major contributors

### **Open Source Benefits**
- **Portfolio building** with cutting-edge AI technology
- **Networking opportunities** with leading researchers
- **Skill development** in distributed systems and AI
- **Impact measurement** through real-world applications

## 📋 Issue Labels & Priority

We use labels to categorize and prioritize issues:

### **Type Labels**
- `bug` - Something isn't working correctly
- `feature` - New functionality or enhancement
- `docs` - Documentation improvements
- `test` - Testing-related issues
- `performance` - Performance optimization opportunities

### **Priority Labels**
- `priority/critical` - Security issues, system breaks
- `priority/high` - Important features, significant bugs
- `priority/medium` - General improvements, minor bugs
- `priority/low` - Nice-to-have features, cosmetic issues

### **Status Labels**
- `good first issue` - Great for newcomers
- `help wanted` - Community contribution welcome
- `in progress` - Someone is actively working on this
- `needs review` - Ready for maintainer review
- `blocked` - Waiting on external dependency

### **Component Labels**
- `component/nwtn` - NWTN Orchestrator
- `component/agents` - Agent Framework
- `component/teachers` - Teacher Models
- `component/safety` - Safety Infrastructure
- `component/p2p` - P2P Federation
- `component/tokenomics` - FTNS Economy
- `component/governance` - Governance System
- `component/improvement` - Self-Improvement
- `component/api` - REST API and WebSocket endpoints
- `component/ui` - Web interface and frontend
- `component/websocket` - Real-time communication features

## 🛡️ Safety & Security

PRSM deals with AI systems and research data, so security is paramount:

### **Security Guidelines**
- **Never commit** API keys, passwords, or sensitive data
- **Follow security** best practices in all code
- **Report security vulnerabilities** privately to maintainers
- **Validate inputs** thoroughly in all functions
- **Use secure communication** for all network operations

### **Safety Considerations**
- **AI safety** - Ensure AI outputs are safe and beneficial
- **Data privacy** - Protect user data and research information
- **Transparent operation** - Maintain auditability in all processes
- **Emergency procedures** - Support for rapid response to issues

## 📞 Getting Help

### **Community Support**
- **GitHub Discussions** - Q&A and general discussion
- **Issue tracker** - Bug reports and feature requests
- **Documentation** - Comprehensive guides and references
- **Code comments** - Extensive inline documentation

### **Direct Contact**
- **Maintainer review** - Tag maintainers in PRs and issues
- **Research collaborations** - Reach out for academic partnerships
- **Enterprise support** - Contact for commercial applications
- **Conference speaking** - Opportunities for project presentation

## 🎉 Thank You!

Every contribution, no matter how small, helps build an open-source commons for AI infrastructure. Whether you fix a typo, add a test, implement a feature, run a node, or publish a dataset — you're helping make the network more valuable for everyone who relies on it.

**Welcome to the PRSM community!**

---

> _"Science advances through collaboration, not competition. Thank you for being part of that advancement."_

**Ready to contribute? Check out our [good first issues](https://github.com/prsm-network/PRSM/labels/good%20first%20issue) to get started!**