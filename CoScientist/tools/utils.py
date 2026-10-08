
def tool(name: str = None, description: str = None):
    def wrapper(func):
        func._is_tool = True
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__
        return func
    return wrapper

def toolset(name: str = None, description: str = None):
    def wrapper(func):
        func._is_toolset = True
        func._tool_name = name or func.__name__
        func._tool_description = description or func.__doc__
        return func
    return wrapper
