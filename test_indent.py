def test_function():
    try:
        if True:
            try:
                pass
            except Exception as e:
                print(f"Error: {e}")

        print("This is part of the try block")
    except Exception as e:
        print(f"Outer error: {e}")

print("Testing syntax...")
