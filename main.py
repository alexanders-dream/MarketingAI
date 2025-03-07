import streamlit as st
from ui import setup_sidebar, initialize_llm, render_file_upload, render_task_interface

def main():
    
    # Main header
    st.title("Marketing Agent Pro")
    st.markdown("---")
    
    # Setup sidebar and get selected task
    task, use_rag = setup_sidebar()
    
    # Initialize LLM client
    llm = initialize_llm()
    
    # Main content area
    with st.container():
        tab_analysis, tab_manual = st.tabs(["📄 Document Analysis", "✍️ Manual Input"])
        
        with tab_analysis:
            render_file_upload(use_rag)
        
        with tab_manual:
            st.info("Coming soon: Direct input without document upload")
        
        if 'extracted_data' in st.session_state:
            with st.expander("Debug: Extracted Data"):
                st.write(st.session_state.extracted_data)
            render_task_interface(llm, task)
        else:
            st.info("✨ Upload a document or use manual input to get started")

if __name__ == "__main__":
    main()