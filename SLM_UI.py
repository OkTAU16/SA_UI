from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.behaviors import DragBehavior
from kivy.core.window import Window
from kivy.graphics import Rectangle, Color, Line, Ellipse
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.checkbox import CheckBox
from kivy.uix.boxlayout import BoxLayout
import os


# from kivy.graphics import Color, Ellipse, Line
# from random import random

class ColorCheckBox(CheckBox):
    def __init__(self, color=(1, 0, 0, 1), **kwargs):
        super(ColorCheckBox, self).__init__(**kwargs)
        self.color = color

        with self.canvas.before:
            self.rect_color = Color(*self.color)
            self.rectangle = Rectangle(pos=self.pos, size=self.size)
            self.marker_color = Color(1, 1, 1, 1)
            self.marker = Ellipse(pos=(self.x + 15, self.y + self.height / 2 - 10), size=(20, 20))

        self.bind(pos=self.update_canvas, size=self.update_canvas)

    def update_canvas(self, *args):
        self.rectangle.pos = self.pos
        self.rectangle.size = self.size
        self.marker.pos = (self.x + 15, self.y + self.height / 2 - 10)

    def on_state(self, instance, value):
        self.marker_color.rgba = [1, 1, 1, 1] if value == 'normal' else self.color
class FileDropApp(App):
    def __init__(self, **kwargs):
        super().__init__()
        self.drop_area_height = None
        self.drop_area_width = None
        self.drop_area_y = None
        self.drop_area_x = None

    def build(self):
        layout = FloatLayout()

        self.label_checkbox = Label(text="CheckBox lookout", size_hint=(None, None), pos_hint={'center_x': 0.9, 'center_y': 0.95})
        self.label_radio = Label(text="radio lookout", size_hint=(None, None),
                                    pos_hint={'center_x': 0.9, 'center_y': 0.8})
        self.checkbox = ColorCheckBox(active=False, size_hint=(None, None), size=(50, 50),
                                      pos_hint={'right': 0.9, 'top': 0.9}, color=(1, 0, 0, 1))
        layout.add_widget(self.label_checkbox)
        layout.add_widget(self.checkbox)
        self.radio1 = ColorCheckBox(group='options', size_hint=(None, None), size=(50, 50), pos_hint={'x': 0.8, 'top': 0.7})

        self.radio2 = ColorCheckBox(group='options', size_hint=(None, None), size=(50, 50), pos_hint={'x': 0.9, 'top': 0.7})
        layout.add_widget(self.radio1)
        layout.add_widget(self.radio2)
        layout.add_widget(self.label_radio)

        self.label_toggle = Label(text="toggle lookout", size_hint=(None, None),
                                    pos_hint={'center_x': 0.9, 'center_y': 0.55})
        self.toggle_button = ToggleButton(text='Toggle', size_hint=(None, None), size=(100, 50), pos_hint={'center_x': 0.9, 'top': 0.4})

        layout.add_widget(self.label_toggle)
        layout.add_widget(self.toggle_button)
        '''
        # Create a CheckBox
        self.checkbox = CheckBox(active=False, size_hint=(None, None), size=(50, 50), pos_hint={'right': 0.9, 'top': 0.9})
        layout.add_widget(self.label_checkbox)
        layout.add_widget(self.checkbox)

        self.label_radio = Label(text="radio lookout", size_hint=(None, None),
                                    pos_hint={'center_x': 0.9, 'center_y': 0.8})
        self.radio1 = CheckBox(group='options', size_hint=(None, None), size=(50, 50), pos_hint={'x': 0.9, 'top': 0.7})

        self.radio2 = CheckBox(group='options', size_hint=(None, None), size=(50, 50), pos_hint={'x': 0.9, 'top': 0.5})
        layout.add_widget(self.radio1)
        layout.add_widget(self.radio2)
        '''
        # Create a label to display dropped file path
        self.file_label = Label(text='Drop files here', size_hint=(None, None),
                                pos_hint={'center_x': 0.25, 'center_y': 0.9})
        self.file_label.bind(size=self.file_label.setter('text_size'))
        layout.add_widget(self.file_label)

        test_button = Button(text='Test', size_hint=(None, None), size=(150, 50),
                               pos_hint={'center_x': 0.15, 'center_y': 0.58})
        test_button.bind(on_press=self.test_dropped_file)
        self.test_submit_label = Label(text='test', size_hint=(None, None),
                                pos_hint={'center_x': 0.25, 'center_y': 0.52})
        self.file_label.bind(size=self.file_label.setter('text_size'))
        layout.add_widget(self.test_submit_label)
        layout.add_widget(test_button)

        submit_button = Button(text='Submit', size_hint=(None, None), size=(150, 50),
                               pos_hint={'center_x': 0.35, 'center_y': 0.58})
        #submit_button.bind(on_press=???)
        layout.add_widget(submit_button)

        next_button = Button(text='Next', size_hint=(None, None), size=(100, 50),
                               pos_hint={'center_x': 0.93, 'center_y': 0.07})
        next_button.bind(on_press=self.next_window)
        layout.add_widget(next_button)

        # Allow the window to receive file drops
        Window.bind(on_drop_file=self.on_drop_file)
        Window.bind(on_resize=self.update_drop_area)

        #new method for defining the dropout area and border
        #self.window_width, self.window_height,self.drop_area_x,self.drop_area_y,self.drop_area_width,self.drop_area_height, line =update_drop_area(Window)
        #self.update_drop_area(Window)


        self.window_width, self.window_height = Window.size
        self.drop_area_x = self.window_width // 16
        self.drop_area_y = 5 * self.window_height // 8
        self.drop_area_width = 3 * self.window_width // 8
        self.drop_area_height = self.window_height // 4
        with layout.canvas:
            Color(1, 1, 1, 1)

            self.line = Line(points=[self.drop_area_x, self.drop_area_y,
                                self.drop_area_x + self.drop_area_width, self.drop_area_y,
                                self.drop_area_x + self.drop_area_width, self.drop_area_y + self.drop_area_height,
                                self.drop_area_x, self.drop_area_y + self.drop_area_height,
                                self.drop_area_x, self.drop_area_y], width=1)

            #Add a label for testing below the drop area
        self.test_label = Label(text='', size_hint=(None, None),
                                pos_hint={'center_x': 0.25, 'center_y': 0.45})
        layout.add_widget(self.test_label)

        return layout
    def update_drop_area(self, Window, *args):

        self.window_width, self.window_height = Window.size
        self.drop_area_x = self.window_width // 16
        self.drop_area_y = 5 * self.window_height // 8
        self.drop_area_width = 3 * self.window_width // 8
        self.drop_area_height = self.window_height // 4
        self.root.canvas.remove(self.line)
        with self.root.canvas:
            Color(1, 1, 1, 1)

            self.line = Line(points=[self.drop_area_x, self.drop_area_y,
                                self.drop_area_x + self.drop_area_width, self.drop_area_y,
                                self.drop_area_x + self.drop_area_width, self.drop_area_y + self.drop_area_height,
                                self.drop_area_x, self.drop_area_y + self.drop_area_height,
                                self.drop_area_x, self.drop_area_y], width=1)

    def on_drop_file(self, window, file_path, x, y):
        # Check if the drop occurred within the specified area
        #if (self.drop_area_x <= x <= self.drop_area_x + self.drop_area_width
        #        and self.drop_area_y <= y <= self.drop_area_y + self.drop_area_height):
        #print("%i", window.left)
        #print("%i", window.width)
        #print("%i", (window.height))
        print("top left corner is (%i,%i)", (self.window_width // 16, self.window_height-self.drop_area_y))
        print("bottom left corner is (%i,%i)", (self.drop_area_x, self.window_height - (self.drop_area_y+self.drop_area_height)))
        print("bottom right corner is (%i,%i)",(self.drop_area_x+self.drop_area_width, self.window_height - (self.drop_area_y + self.drop_area_height)))
        print("top right corner is (%i,%i)", (self.drop_area_x + self.drop_area_width, self.window_height -  self.drop_area_height))
        print("real x:", (x))
        print(y)

        if (self.window_width // 16 <= x//(2/3) <= self.window_width // 16 + (3*self.window_width // 8)
               and self.window_height // 8 <= y//(2/3) <= (3 * self.window_height) // 8):
            print(x)
            print(y)
            # Handle the dropped file
            file_path = file_path.decode('utf-8')  # Convert bytes to string
            self.file_label.text = f'Dropped file: {file_path}'
            self.test_label.text = 'File dropped!'
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

    def test_dropped_file(self, file_path):

        """
        test that all files are of known format
            inputs:
                file_path: dropped file path
            outputs:
        """
        self.fill_all_data()
        pass

    def fill_all_data(self,button=None):
        """
        making sure that all user data and buttons are pressed
        """


        pass

    def next_window(self, *args):
        """
        moving to the next window, and showing the results
        """

        pass

if __name__ == '__main__':
    FileDropApp().run()
