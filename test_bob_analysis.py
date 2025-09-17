import streamlit as st
import sys
import os

# Add pages directory to path
sys.path.append('pages')

def test_bob_analysis():
    st.title("Bob Analysis Test")
    
    try:
        # Test 1: Basic import
        st.write("### Test 1: Importing bob_analysis module")
        from bob_analysis import show_bob_analysis
        st.success("✅ Successfully imported bob_analysis module")
        
        # Test 2: Check if function exists
        st.write("### Test 2: Checking show_bob_analysis function")
        if hasattr(show_bob_analysis, '__call__'):
            st.success("✅ show_bob_analysis function exists and is callable")
        else:
            st.error("❌ show_bob_analysis is not callable")
            return
        
        # Test 3: Try to call the function with error handling
        st.write("### Test 3: Attempting to call show_bob_analysis")
        try:
            show_bob_analysis()
            st.success("✅ show_bob_analysis executed successfully")
        except Exception as e:
            st.error(f"❌ Error calling show_bob_analysis: {str(e)}")
            st.error("Full error details:")
            import traceback
            st.code(traceback.format_exc())
            
    except ImportError as e:
        st.error(f"❌ Import error: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        import traceback
        st.error("Full error details:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    test_bob_analysis() 