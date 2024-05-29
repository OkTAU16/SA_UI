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
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
# from SLM_UI_graphs_window import *
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
        self.cluster_num = None
        self.CV_num = None
        self.target_num = None

    def build(self):
        layout = FloatLayout()
        """
        playing around. not for usage
        
        
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
        """
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
        self.label_time_series = Label(text="Added Time Series", size_hint=(None, None),
                                    pos_hint={'center_x': 0.7, 'center_y': 0.9})
        #self.label_time_series_Yes = Label(text="Yes", size_hint=(None, None),
        #                            pos_hint={'center_x': 0.65, 'center_y': 0.85})
        #self.label_time_series_No = Label(text="No", size_hint=(None, None),
        #                            pos_hint={'center_x': 0.75, 'center_y': 0.85})
        layout.add_widget(self.label_time_series)
        #layout.add_widget(self.label_time_series_Yes)
        #layout.add_widget(self.label_time_series_No)
        self.button_time_series_yes = ColorCheckBox(group='time_series', size_hint=(None, None), color=(0, 1, 1, 1), size=(50, 50), pos_hint={'x': 0.63, 'top': 0.87})

        self.button_time_series_no = ColorCheckBox(group='time_series', size_hint=(None, None), size=(50, 50), pos_hint={'x': 0.73, 'top': 0.87})
        layout.add_widget(self.button_time_series_yes)
        layout.add_widget(self.button_time_series_no)

        self.label_PCA = Label(text="PCA", size_hint=(None, None),
                                    pos_hint={'center_x': 0.7, 'center_y': 0.77})
        layout.add_widget(self.label_PCA)

        self.button_PCA_yes = ColorCheckBox(group='PCA', size_hint=(None, None), color=(0, 1, 1, 1), size=(50, 50), pos_hint={'x': 0.63, 'top': 0.74})

        self.button_PCA_no = ColorCheckBox(group='PCA', size_hint=(None, None), size=(50, 50), pos_hint={'x': 0.73, 'top': 0.74})
        layout.add_widget(self.button_PCA_yes)
        layout.add_widget(self.button_PCA_no)
        self.button_PCA_yes.bind(on_press=self.PCA_show)
        self.button_PCA_no.bind(on_press=self.PCA_show)

        self.PCA_input = TextInput(hint_text="Enter number of clusters", font_size=16, input_filter='int',multiline=True,
                                    size_hint=(0.1, 0.06),pos_hint={'x': 0.79, 'top': 0.74})
        self.button_PCA_submit = Button(text="✓", size_hint=(None, None), font_name='DejaVuSans.ttf', size=(53, 53), pos_hint={'x': 0.9, 'top': 0.74})
        layout.add_widget(self.PCA_input)
        layout.add_widget(self.button_PCA_submit)
        self.PCA_input.opacity = 0
        self.button_PCA_submit.opacity = 0
        self.button_PCA_submit.bind(on_press=self.PCA_submit)
        self.PCA_input.bind(text=self.text_size_change)




        self.label_CV = Label(text="CV", size_hint=(None, None),
                                    pos_hint={'center_x': 0.7, 'center_y': 0.64})
        layout.add_widget(self.label_CV)

        self.button_CV_yes = ColorCheckBox(group='CV', size_hint=(None, None), color=(0, 1, 1, 1), size=(50, 50), pos_hint={'x': 0.63, 'top': 0.60})

        self.button_CV_no = ColorCheckBox(group='CV', size_hint=(None, None), size=(50, 50), pos_hint={'x': 0.73, 'top': 0.60})
        layout.add_widget(self.button_CV_yes)
        layout.add_widget(self.button_CV_no)
        self.button_CV_yes.bind(on_press=self.CV_show)
        self.button_CV_no.bind(on_press=self.CV_show)

        self.CV_input = TextInput(hint_text="Enter number of ???", font_size=16, input_filter='int',multiline=True,
                                    size_hint=(0.1, 0.06),pos_hint={'x': 0.79, 'top': 0.60})
        self.button_CV_submit = Button(text="✓", size_hint=(None, None), font_name='DejaVuSans.ttf', size=(53, 53), pos_hint={'x': 0.9, 'top': 0.60})
        layout.add_widget(self.CV_input)
        layout.add_widget(self.button_CV_submit)
        self.CV_input.opacity = 0
        self.button_CV_submit.opacity = 0
        self.button_CV_submit.bind(on_press=self.CV_submit)
        self.CV_input.bind(text=self.text_size_change)

        self.label_targets = Label(text="Number of targets", size_hint=(None, None),
                                    pos_hint={'center_x': 0.7, 'center_y': 0.5})
        layout.add_widget(self.label_targets)
        self.target_input = TextInput(hint_text="Enter number of parameters", font_size=15, input_filter='int',multiline=True,
                                    size_hint=(0.09, 0.06),pos_hint={'x': 0.63, 'top': 0.46})
        self.button_targets_submit = Button(text="✓", size_hint=(None, None), font_name='DejaVuSans.ttf', size=(53, 53), pos_hint={'x': 0.73, 'top': 0.46})
        layout.add_widget(self.target_input)
        layout.add_widget(self.button_targets_submit)
        self.target_input.bind(text=self.text_size_change)
        self.button_targets_submit.bind(on_press=self.target_submit)


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

    def PCA_show(self, instance):
        # Show or hide TextInput and V button based on selected radio button
        if instance == self.button_PCA_yes and instance.state == 'down':
            self.PCA_input.opacity = 1
            self.button_PCA_submit.opacity = 1
        elif instance == self.button_PCA_no and instance.state == 'down':
            self.PCA_input.opacity = 0
            self.button_PCA_submit.opacity = 0
            self.cluster_num = None
            self.PCA_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            self.PCA_input.background_color = (1, 1, 1, 1)
            self.PCA_input.text = ''
            self.PCA_input.font_size = 16
        else:
            self.PCA_input.opacity = 0
            self.button_PCA_submit.opacity = 0
            self.cluster_num = None
            self.PCA_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            self.PCA_input.background_color = (1, 1, 1, 1)
            self.PCA_input.text = ''
            self.PCA_input.font_size = 16

    def CV_show(self, instance):
        # Show or hide TextInput and V button based on selected radio button
        if instance == self.button_CV_yes and instance.state == 'down':
            self.CV_input.opacity = 1
            self.button_CV_submit.opacity = 1
        elif instance == self.button_CV_no and instance.state == 'down':
            self.CV_input.opacity = 0
            self.button_CV_submit.opacity = 0
            self.CV_num = None
            self.CV_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            self.CV_input.background_color = (1, 1, 1, 1)
            self.CV_input.text = ''
            self.CV_input.font_size = 16
        else:
            self.CV_input.opacity = 0
            self.button_CV_submit.opacity = 0
            self.CV_num = None
            self.CV_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            self.CV_input.background_color = (1, 1, 1, 1)
            self.CV_input.text = ''
            self.CV_input.font_size = 16

    def PCA_submit(self, instance):
        user_input = self.PCA_input.text
        if user_input:
            self.cluster_num = user_input
            print(self.cluster_num)
            self.PCA_input.foreground_color = (0, 0.6, 0, 1)  # RGBA for green
            self.PCA_input.background_color = (0.6, 1, 0.9, 1)
        else:
            self.cluster_num = None
            print(self.cluster_num)
            self.PCA_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color

    def CV_submit(self, instance):
        user_input = self.CV_input.text
        if user_input:
            self.CV_num = user_input
            print(self.CV_num)
            self.CV_input.foreground_color = (0, 0.6, 0, 1)  # RGBA for green
            self.CV_input.background_color = (0.6, 1, 0.9, 1)
        else:
            self.CV_num = None
            print(self.CV_num)
            self.CV_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            self.CV_input.background_color = (1, 1, 1, 1)

    def target_submit(self, instance):
        user_input = self.target_input.text
        if user_input:
            self.target_num = user_input
            print(self.target_num)
            self.target_input.foreground_color = (0, 0.6, 0, 1)  # RGBA for green
            self.target_input.background_color = (0.6, 1, 0.9, 1)
        else:
            self.target_num = None
            print(self.target_num)
            self.target_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            self.target_input.background_color = (1, 1, 1, 1)

    def text_size_change(self, instance, value):
        print(f"Text changed to: {value}")  # Debug statement
        if value:
            instance.font_size = 28  # Larger font size for user input

        else:
            instance.font_size = 8  # same size as before

if __name__ == '__main__':
    FileDropApp().run()
