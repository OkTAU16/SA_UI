from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.behaviors import DragBehavior
from kivy.core.window import Window
from kivy.graphics import Rectangle, Color, Line
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.filechooser import FileChooserIconView
import os

# from kivy.graphics import Color, Ellipse, Line
# from random import random
'''
class DraggableTextInput(DragBehavior, TextInput):
    pass


class SLMUiAPP(App):
    def build(self):
        layout = FloatLayout()

        # Create a button with specific size and position
        next_button = Button(text='next',
                             size_hint=(None, None),  # Disable automatic size adjustment
                             size=(200, 100))  # Set button size (width, height)
        # Set button position (x, y)
        next_button.pos_hint = {'right': 1, 'bottom': 1}
        draggable_text_input = DraggableTextInput(
                               size_hint=(None, None),  # Disable automatic size adjustment
                               size=(300, 50),          # Set text input size (width, height)
                               pos=(100, 200))
        draggable_text_input.id = 'my_text_input'

        text_input = TextInput(
            size_hint=(None, None),  # Disable automatic size adjustment
            size=(300, 50),  # Set text input size (width, height)
            pos=(100, 200))
        text_input.id = 'my_text_input'

        title_label = Label(text='Title',
                            size_hint=(None, None),  # Disable automatic size adjustment
                            size=(300, 50),  # Set label size (width, height)
                            pos=(100, 300))

        submit_button = Button(text='Submit',
                               size_hint=(None, None),  # Disable automatic size adjustment
                               size=(200, 50),  # Set button size (width, height)
                               pos=(150, 100))  # Set button position (x, y)

        # Bind the save_input_data function to the on_press event of the submit button
        submit_button.bind(on_press=self.save_input_data)
        # Add the button to the layout
        layout.add_widget(next_button)
        layout.add_widget(title_label)
        layout.add_widget(draggable_text_input)
        #layout.add_widget(text_input)
        layout.add_widget(submit_button)

        return layout

    def save_input_data(self, instance):
        # Access the TextInput widget using its id
        text_input = self.root.ids.layout.ids.my_text_input
        input_data = text_input.text
        print("Input data:", input_data)

    # Bind the function to the on_text_validate event


if __name__ == '__main__':
    SLMUiAPP().run()
'''


class FileDropApp(App):
    def __init__(self, **kwargs):
        super().__init__()
        self.drop_area_height = None
        self.drop_area_width = None
        self.drop_area_y = None
        self.drop_area_x = None

    def build(self):
        layout = FloatLayout()

        # Create a label to display dropped file path
        self.file_label = Label(text='Drop files in the box below, or select manually', size_hint=(None, None),
                                pos_hint={'center_x': 0.25, 'center_y': 0.9})
        self.file_label.bind(size=self.file_label.setter('text_size'))
        layout.add_widget(self.file_label)

        # Allow the window to receive file drops
        Window.bind(on_drop_file=self.on_drop_file)
        self.window_width, self.window_height = Window.size
        self.drop_area_x = self.window_width // 16
        self.drop_area_y = 5 * self.window_height // 8
        self.drop_area_width = 3 * self.window_width // 8
        self.drop_area_height = self.window_height // 4
        with layout.canvas:
            Color(1, 1, 1, 1)

            line = Line(points=[self.drop_area_x, self.drop_area_y,
                                self.drop_area_x + self.drop_area_width, self.drop_area_y,
                                self.drop_area_x + self.drop_area_width, self.drop_area_y + self.drop_area_height,
                                self.drop_area_x, self.drop_area_y + self.drop_area_height,
                                self.drop_area_x, self.drop_area_y], width=1)

        return layout

    def on_drop_file(self, window, file_path, x, y):
        # Check if the drop occurred within the specified area
        #if (self.drop_area_x <= x <= self.drop_area_x + self.drop_area_width
        #        and self.drop_area_y <= y <= self.drop_area_y + self.drop_area_height):
        print(x)
        print(y)
        if (self.drop_area_x <= x <= self.drop_area_x + self.drop_area_width
                and self.window_height // 8 <= y <= 5 * self.window_height // 8):
            print(x)
            print(y)
            # Handle the dropped file
            file_path = file_path.decode('utf-8')  # Convert bytes to string
            self.file_label.text = f'Dropped file: {file_path}'
            EXTENTIONS = (".xlsx", ".xlsm", ".xltx", ".xltm")
            print(file_path.endswith(EXTENTIONS))
            print(os.path.isfile(file_path))
            # Call a function to process the dropped file
            self.process_dropped_file(file_path)
        else:
            self.file_label.text = 'Drop files only within the specified area!'

    def process_dropped_file(self, file_path):
        # Implement your file processing logic here
        print(f"Processing file: {file_path}")


if __name__ == '__main__':
    FileDropApp().run()
