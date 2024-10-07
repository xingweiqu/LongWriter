import json

def truncate_text(text, max_length=50):
    return (text[:max_length] + '...') if len(text) > max_length else text

def read_and_display_json(file_path, max_items=3, max_element_length=50):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        if isinstance(data, list):
            print(f"Total number of items in the list: {len(data)}")
            
            for index, item in enumerate(data[:max_items]):
                if isinstance(item, dict):
                    print(f"\n[{index}]: dict (Number of keys: {len(item)})")
                    for key, value in item.items():
                        value_type = type(value).__name__
                        if key == 'content' and isinstance(value, list):
                            print(f"[{index}].{key}: list (Length: {len(value)})")
                            print("Content (truncated):")
                            for i, content_item in enumerate(value):
                                truncated_item = truncate_text(content_item, max_element_length)
                                print(f"  [{index}].content[{i}]: {truncated_item}")
                        else:
                            print(f"[{index}].{key}: {value} (type: {value_type})")
                else:
                    print(f"\n[{index}]: {type(item).__name__}")
                
                if index >= max_items - 1:
                    break
        else:
            print("The JSON file does not contain a list at the top level.")
    
    except json.JSONDecodeError:
        print("Error: Invalid JSON file")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    file_path = "Novels_from_enread_picked_in_sections.json"
    read_and_display_json(file_path)