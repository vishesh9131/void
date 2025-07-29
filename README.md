# VS Aware 🧠

**The Machine Learning Engineer's Code Editor**

VS Aware is a specialized version of VSCode designed specifically for machine learning engineers, data scientists, and AI developers. Built on the foundation of Void (VSCode fork), VS Aware provides intelligent ML-specific tools and workflows that streamline the entire machine learning development lifecycle.

## 🚀 Key Features

### 🤖 Intelligent AutoML Integration
- **Right-click Dataset Analysis**: Instantly analyze CSV, JSON, or Parquet files
- **Smart Model Recommendations**: AI-powered suggestions for optimal model architectures
- **Framework-Agnostic Code Generation**: Generate production-ready code for PyTorch, TensorFlow, or scikit-learn
- **Hyperparameter Optimization**: Built-in grid search, random search, and Bayesian optimization
- **Data Profiling Reports**: Comprehensive statistical analysis and data quality insights

### 📊 GPU Resource Dashboard
- **Real-time Monitoring**: Live GPU/TPU utilization, memory usage, and temperature tracking
- **Process Management**: View and manage GPU processes with memory attribution
- **Smart Alerts**: Configurable warnings for memory, temperature, and power thresholds
- **Historical Tracking**: Performance charts and resource usage trends
- **Multi-GPU Support**: Monitor multiple GPUs simultaneously

### 🔄 Model & Dataset Version Control
- **Native DVC Integration**: Seamless Data Version Control for datasets and models
- **Git-LFS Support**: Automatic large file handling for model weights
- **Visual Diffs**: Compare model architectures, weights, and performance metrics
- **Experiment Tracking**: Link models to datasets with full lineage tracking
- **Statistical Comparisons**: Histogram comparisons for weight distributions

### 🔍 Tensor & DataFrame Visualizer
- **Hover Inspection**: Instant tensor shape, dtype, and statistics on hover
- **Interactive Visualizations**: Heatmaps, histograms, and dimensionality reduction plots
- **DataFrame Explorer**: Column statistics, correlation matrices, and data previews
- **Memory Usage Tracking**: Real-time memory consumption monitoring
- **Multi-Framework Support**: Works with NumPy, PyTorch, TensorFlow, and Pandas

### 🐛 ML Pipeline Debugger
- **Gradient Analysis**: Detect vanishing/exploding gradients during training
- **NaN Detection**: Automatic detection of NaN values in tensor operations
- **Data Pipeline Profiling**: Identify bottlenecks in data loading and preprocessing
- **Timeline Profiling**: Detailed execution timelines for training loops
- **Custom Breakpoints**: ML-specific debugging with tensor inspection

### 📈 Experiment Tracking Hub
- **Live Metrics Dashboard**: Real-time loss, accuracy, and custom metric visualization
- **Experiment Comparison**: Side-by-side hyperparameter and performance analysis
- **Integration Ready**: One-click setup for TensorBoard, Weights & Biases, MLflow
- **Collaboration Tools**: Share experiments and results with team members
- **Automated Logging**: Capture metrics without code changes

### 📓 Notebook ⇄ Production Converter
- **AI-Assisted Refactoring**: Convert Jupyter notebooks to production Python modules
- **Automatic Test Generation**: Create unit tests from notebook cells
- **Dependency Management**: Smart requirements.txt generation
- **Code Quality**: Automatic formatting and best practices enforcement
- **Modular Architecture**: Extract functions and classes from notebook cells

### 🧠 Domain-Specific IntelliSense
- **Tensor Shape Awareness**: Smart autocompletion with shape propagation
- **ML Design Patterns**: Templates for transfer learning, custom layers, and more
- **Cloud ML API Integration**: Autocompletion for AWS SageMaker, Google Vertex AI, Azure ML
- **Framework Documentation**: Inline docs for PyTorch, TensorFlow, and scikit-learn
- **Best Practices**: Suggestions for optimal ML code patterns

### 🏷️ Data Annotation Suite
- **Multi-Modal Support**: Image, audio, and text labeling tools
- **Team Collaboration**: Real-time collaborative annotation
- **Auto-Labeling**: Model-assisted pre-tagging and suggestions
- **Quality Control**: Annotation validation and inter-annotator agreement
- **Export Options**: Multiple format support (COCO, YOLO, etc.)

### 🚀 Model Deployment Workflows
- **One-Click Containerization**: Automatic Dockerfile generation for models
- **Cloud Provisioning**: Deploy to AWS, GCP, Azure with guided wizards
- **Monitoring Integration**: Connect to Prometheus, Grafana dashboards
- **API Generation**: Automatic REST API creation for model serving
- **CI/CD Integration**: GitHub Actions and MLOps pipeline templates

## 🛠️ Installation

### Prerequisites
- Node.js 20.x or later
- Python 3.8+ (for ML features)
- Git (for version control features)
- Optional: CUDA drivers (for GPU monitoring)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/vsaware/vs-aware.git
cd vs-aware

# Install dependencies
npm install

# Build the application
npm run compile

# Start VS Aware
npm run electron
```

### Package Installation
```bash
# Install ML dependencies
pip install torch tensorflow scikit-learn pandas numpy matplotlib seaborn
pip install dvc tensorboard wandb mlflow

# Optional: GPU monitoring tools
pip install pynvml gpustat
```

## 🎯 Getting Started

### 1. Analyze Your First Dataset
1. Open VS Aware
2. Right-click any `.csv`, `.json`, or `.parquet` file
3. Select "Generate Model Prototype"
4. Choose your preferred ML framework
5. Review generated code and recommendations

### 2. Monitor GPU Resources
1. Open Command Palette (`Ctrl+Shift+P`)
2. Run "VS Aware: Open GPU Dashboard"
3. Start monitoring to see real-time resource usage
4. Configure alerts for your workflow

### 3. Track Experiments
1. Open the VS Aware sidebar
2. Navigate to "Experiment Tracking"
3. Connect your preferred tracking service
4. Start logging metrics automatically

## 📚 Documentation

- **[User Guide](docs/user-guide.md)**: Complete feature documentation
- **[API Reference](docs/api.md)**: Extension API for developers
- **[ML Workflows](docs/workflows.md)**: Best practices and examples
- **[Configuration](docs/configuration.md)**: Customization options

## 🤝 Contributing

We welcome contributions from the ML community! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
npm install

# Start development build
npm run watch

# Run tests
npm test
```

## 📦 ML-Specific Dependencies

VS Aware includes optimized support for:

- **Deep Learning**: PyTorch, TensorFlow, JAX, Hugging Face Transformers
- **Classical ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Data Processing**: Pandas, NumPy, Polars, Dask
- **Visualization**: Matplotlib, Seaborn, Plotly, Bokeh
- **Experiment Tracking**: MLflow, Weights & Biases, TensorBoard, Neptune
- **Model Serving**: FastAPI, Flask, TorchServe, TensorFlow Serving
- **Cloud Platforms**: AWS SageMaker, Google Vertex AI, Azure ML

## 🎉 What Makes VS Aware Special?

Unlike general-purpose editors, VS Aware is built specifically for ML workflows:

- **Zero Configuration**: ML tools work out of the box
- **Intelligent Defaults**: Optimized settings for ML development
- **Framework Agnostic**: Works with any ML framework or library
- **Performance Focused**: Optimized for large datasets and models
- **Collaboration Ready**: Built-in tools for team ML projects

## 📄 License

VS Aware is licensed under the [MIT License](LICENSE.txt), same as the original VSCode project.

## 🙏 Acknowledgments

VS Aware is built upon the excellent work of:
- [Microsoft VSCode](https://github.com/microsoft/vscode) - The foundation
- [Void Editor](https://github.com/voideditor/void) - The starting point
- The entire open-source ML community

---

**Ready to supercharge your ML development?** [Download VS Aware](https://github.com/vsaware/vs-aware/releases) today! 🚀
