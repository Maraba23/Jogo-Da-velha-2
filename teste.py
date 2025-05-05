import win32gui

def enum_windows_callback(hwnd, extra):
    if win32gui.IsWindowVisible(hwnd):
        title = win32gui.GetWindowText(hwnd)
        class_name = win32gui.GetClassName(hwnd)
        if title or class_name:
            print(f"HWND: {hwnd} | Class: {class_name} | Title: {title}")

win32gui.EnumWindows(enum_windows_callback, None)

# import win32gui
# import win32con

# def check_window(hwnd):
#     if not win32gui.IsWindowVisible(hwnd):
#         return

#     title = win32gui.GetWindowText(hwnd)
#     class_name = win32gui.GetClassName(hwnd)
#     style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)

#     if (style & win32con.WS_EX_LAYERED) and (style & win32con.WS_EX_TRANSPARENT):
#         print(f"[POTENCIAL] HWND: {hwnd} | Class: {class_name} | Title: {title}")
#     else:
#         print(f"[X] HWND: {hwnd} | Class: {class_name} | Title: {title}")

# win32gui.EnumWindows(check_window, None)
