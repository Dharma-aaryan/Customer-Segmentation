import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import Dict, List, Any
import base64

class NotebookAnalyzer:
    """Analyzes Jupyter notebooks and extracts insights."""
    
    def __init__(self, notebook_content: Dict):
        self.notebook = notebook_content
        self.cells = notebook_content.get('cells', [])
        self.metadata = notebook_content.get('metadata', {})
    
    def show_overview(self):
        """Display notebook overview and basic statistics."""
        st.subheader("📋 Notebook Overview")
        
        # Basic statistics
        code_cells = [cell for cell in self.cells if cell.get('cell_type') == 'code']
        markdown_cells = [cell for cell in self.cells if cell.get('cell_type') == 'markdown']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cells", len(self.cells))
        with col2:
            st.metric("Code Cells", len(code_cells))
        with col3:
            st.metric("Markdown Cells", len(markdown_cells))
        with col4:
            executed_cells = len([cell for cell in code_cells if cell.get('execution_count')])
            st.metric("Executed Cells", executed_cells)
        
        # Notebook metadata
        st.subheader("📝 Notebook Metadata")
        if self.metadata:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'kernelspec' in self.metadata:
                    kernel_info = self.metadata['kernelspec']
                    st.write("**Kernel Information:**")
                    st.json(kernel_info)
            
            with col2:
                if 'language_info' in self.metadata:
                    lang_info = self.metadata['language_info']
                    st.write("**Language Information:**")
                    st.json(lang_info)
        
        # Cell execution timeline
        if code_cells:
            st.subheader("⏱️ Execution Timeline")
            execution_data = []
            
            for i, cell in enumerate(code_cells):
                exec_count = cell.get('execution_count')
                if exec_count:
                    execution_data.append({
                        'Cell Index': i,
                        'Execution Count': exec_count,
                        'Cell Number': i + 1
                    })
            
            if execution_data:
                exec_df = pd.DataFrame(execution_data)
                fig = px.line(
                    exec_df, x='Cell Index', y='Execution Count',
                    title="Cell Execution Order",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No execution data found in the notebook.")
        
        # Cell length distribution
        st.subheader("📊 Cell Length Distribution")
        cell_lengths = []
        cell_types = []
        
        for cell in self.cells:
            source = cell.get('source', [])
            if isinstance(source, list):
                length = sum(len(line) for line in source)
            else:
                length = len(source)
            
            cell_lengths.append(length)
            cell_types.append(cell.get('cell_type', 'unknown'))
        
        length_df = pd.DataFrame({
            'Cell Type': cell_types,
            'Length': cell_lengths
        })
        
        fig = px.box(
            length_df, x='Cell Type', y='Length',
            title="Cell Length Distribution by Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def show_code_analysis(self):
        """Analyze code content and patterns."""
        st.subheader("💻 Code Analysis")
        
        code_cells = [cell for cell in self.cells if cell.get('cell_type') == 'code']
        
        if not code_cells:
            st.warning("No code cells found in the notebook.")
            return
        
        # Extract all code
        all_code = []
        for cell in code_cells:
            source = cell.get('source', [])
            if isinstance(source, list):
                all_code.extend(source)
            else:
                all_code.append(source)
        
        code_text = '\n'.join(all_code)
        
        # Library imports analysis
        st.subheader("📚 Library Imports")
        imports = self._extract_imports(code_text)
        
        if imports:
            import_df = pd.DataFrame({
                'Library': list(imports.keys()),
                'Usage Count': list(imports.values())
            }).sort_values('Usage Count', ascending=False)
            
            fig = px.bar(
                import_df.head(15), x='Usage Count', y='Library',
                orientation='h',
                title="Most Used Libraries"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show import statements
            with st.expander("View Import Statements"):
                import_lines = [line for line in all_code if 'import' in line and not line.strip().startswith('#')]
                for line in import_lines[:20]:  # Show first 20 imports
                    st.code(line.strip())
        
        # Function definitions
        st.subheader("🔧 Function Definitions")
        functions = self._extract_functions(code_text)
        
        if functions:
            st.info(f"Found {len(functions)} function definitions")
            
            with st.expander("View Function Definitions"):
                for func in functions:
                    st.code(func, language='python')
        
        # ML/Data Science patterns
        st.subheader("🤖 ML/Data Science Patterns")
        patterns = self._analyze_ml_patterns(code_text)
        
        if patterns:
            pattern_df = pd.DataFrame({
                'Pattern': list(patterns.keys()),
                'Occurrences': list(patterns.values())
            }).sort_values('Occurrences', ascending=False)
            
            fig = px.bar(
                pattern_df, x='Occurrences', y='Pattern',
                orientation='h',
                title="ML/Data Science Patterns Found"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Code complexity metrics
        st.subheader("📊 Code Complexity")
        complexity_metrics = self._calculate_complexity_metrics(code_cells)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Lines per Cell", f"{complexity_metrics['avg_lines']:.1f}")
        with col2:
            st.metric("Max Lines in Cell", complexity_metrics['max_lines'])
        with col3:
            st.metric("Total Lines of Code", complexity_metrics['total_lines'])
    
    def _extract_imports(self, code_text: str) -> Dict[str, int]:
        """Extract and count library imports."""
        import_pattern = r'(?:from\s+(\w+)|import\s+(\w+))'
        matches = re.findall(import_pattern, code_text, re.MULTILINE)
        
        imports = {}
        for match in matches:
            lib = match[0] or match[1]
            if lib and not lib.startswith('_'):
                imports[lib] = imports.get(lib, 0) + 1
        
        return imports
    
    def _extract_functions(self, code_text: str) -> List[str]:
        """Extract function definitions from code."""
        function_pattern = r'def\s+\w+\([^)]*\):[^\n]*(?:\n(?:\s{4}.*|\s*\n))*'
        functions = re.findall(function_pattern, code_text, re.MULTILINE)
        return functions
    
    def _analyze_ml_patterns(self, code_text: str) -> Dict[str, int]:
        """Analyze ML and data science patterns in code."""
        patterns = {
            'Data Loading': len(re.findall(r'(?:read_csv|read_json|read_excel|pd\.read)', code_text)),
            'Data Preprocessing': len(re.findall(r'(?:fillna|dropna|drop_duplicates|StandardScaler|MinMaxScaler)', code_text)),
            'Model Training': len(re.findall(r'(?:\.fit\(|train_test_split|cross_val)', code_text)),
            'Model Evaluation': len(re.findall(r'(?:accuracy_score|precision_score|recall_score|f1_score|confusion_matrix)', code_text)),
            'Visualization': len(re.findall(r'(?:plt\.|sns\.|plotly|px\.)', code_text)),
            'Feature Engineering': len(re.findall(r'(?:transform|fit_transform|OneHotEncoder|LabelEncoder)', code_text))
        }
        
        return {k: v for k, v in patterns.items() if v > 0}
    
    def _calculate_complexity_metrics(self, code_cells: List[Dict]) -> Dict[str, Any]:
        """Calculate code complexity metrics."""
        line_counts = []
        
        for cell in code_cells:
            source = cell.get('source', [])
            if isinstance(source, list):
                line_count = len([line for line in source if line.strip()])
            else:
                line_count = len([line for line in source.split('\n') if line.strip()])
            line_counts.append(line_count)
        
        return {
            'avg_lines': np.mean(line_counts) if line_counts else 0,
            'max_lines': max(line_counts) if line_counts else 0,
            'total_lines': sum(line_counts)
        }
    
    def show_visualizations(self):
        """Display visualizations found in the notebook."""
        st.subheader("📊 Notebook Visualizations")
        
        # Look for cells with outputs containing plots
        plot_cells = []
        for i, cell in enumerate(self.cells):
            if cell.get('cell_type') == 'code':
                outputs = cell.get('outputs', [])
                source = cell.get('source', [])
                
                # Check if cell contains plotting code
                if isinstance(source, list):
                    code = '\n'.join(source)
                else:
                    code = source
                
                has_plot = any(keyword in code.lower() for keyword in [
                    'plt.', 'sns.', 'plotly', 'px.', '.plot(', 'matplotlib'
                ])
                
                if has_plot or any('image' in str(output) for output in outputs):
                    plot_cells.append((i, cell, code))
        
        if not plot_cells:
            st.info("No visualization cells found in the notebook.")
            return
        
        st.success(f"Found {len(plot_cells)} cells with visualizations")
        
        # Display plot cells
        for i, (cell_idx, cell, code) in enumerate(plot_cells):
            with st.expander(f"📊 Visualization Cell {cell_idx + 1}"):
                st.code(code, language='python')
                
                # Try to display any image outputs
                outputs = cell.get('outputs', [])
                for output in outputs:
                    if output.get('output_type') == 'display_data':
                        data = output.get('data', {})
                        
                        # Check for image data
                        if 'image/png' in data:
                            try:
                                img_data = data['image/png']
                                if isinstance(img_data, list):
                                    img_data = ''.join(img_data)
                                
                                # Decode base64 image
                                img_bytes = base64.b64decode(img_data)
                                st.image(img_bytes, caption=f"Output from Cell {cell_idx + 1}")
                            except Exception as e:
                                st.error(f"Could not display image: {str(e)}")
                        
                        # Check for text/html (for plotly plots)
                        elif 'text/html' in data:
                            html_content = data['text/html']
                            if isinstance(html_content, list):
                                html_content = ''.join(html_content)
                            
                            # Only show if it's not too long
                            if len(html_content) < 10000:
                                with st.container():
                                    st.write("HTML Output (from Plotly or similar):", unsafe_allow_html=True)
                                    st.markdown(html_content[:1000] + "..." if len(html_content) > 1000 else html_content, unsafe_allow_html=True)
        
        # Visualization pattern analysis
        st.subheader("📈 Visualization Analysis")
        
        all_plot_code = '\n'.join([code for _, _, code in plot_cells])
        
        viz_patterns = {
            'Matplotlib': len(re.findall(r'plt\.(?:plot|scatter|hist|bar|pie|boxplot)', all_plot_code)),
            'Seaborn': len(re.findall(r'sns\.(?:scatterplot|barplot|histplot|boxplot|heatmap)', all_plot_code)),
            'Plotly': len(re.findall(r'(?:px\.|go\.)(?:scatter|bar|histogram|box|pie)', all_plot_code)),
            'Pandas Plotting': len(re.findall(r'\.plot\(', all_plot_code))
        }
        
        viz_patterns = {k: v for k, v in viz_patterns.items() if v > 0}
        
        if viz_patterns:
            viz_df = pd.DataFrame({
                'Library': list(viz_patterns.keys()),
                'Usage Count': list(viz_patterns.values())
            })
            
            fig = px.pie(
                viz_df, values='Usage Count', names='Library',
                title="Visualization Libraries Usage"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cell execution status for plot cells
        st.subheader("⚡ Plot Cell Execution Status")
        exec_status = []
        
        for cell_idx, cell, _ in plot_cells:
            exec_count = cell.get('execution_count')
            has_output = bool(cell.get('outputs'))
            
            exec_status.append({
                'Cell': f"Cell {cell_idx + 1}",
                'Executed': 'Yes' if exec_count else 'No',
                'Has Output': 'Yes' if has_output else 'No',
                'Execution Count': exec_count or 0
            })
        
        if exec_status:
            status_df = pd.DataFrame(exec_status)
            st.dataframe(status_df, use_container_width=True)
