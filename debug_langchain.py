import importlib
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import langchain
    print(f"langchain version: {langchain.__version__}")
    print(f"langchain path: {langchain.__path__}")
    
    try:
        import langchain.chains
        print("Successfully imported langchain.chains")
        print(f"langchain.chains path: {langchain.chains.__path__}")
        print(f"Available attributes in langchain.chains: {dir(langchain.chains)}")
    except ImportError as e:
        print(f"Failed to import langchain.chains: {e}")

    try:
        from langchain.chains import create_retrieval_chain
        print("Successfully imported create_retrieval_chain from langchain.chains")
    except ImportError as e:
        print(f"Failed to import create_retrieval_chain from langchain.chains: {e}")
        
    try:
        from langchain.chains.retrieval import create_retrieval_chain
        print("Successfully imported create_retrieval_chain from langchain.chains.retrieval")
    except ImportError as e:
        print(f"Failed to import create_retrieval_chain from langchain.chains.retrieval: {e}")

except ImportError as e:
    print(f"Failed to import langchain: {e}")

try:
    import langchain_community
    print(f"langchain_community version: {langchain_community.__version__}")
except ImportError:
    print("langchain_community not installed")

try:
    import langchain_core
    print(f"langchain_core version: {langchain_core.__version__}")
except ImportError:
    print("langchain_core not installed")
