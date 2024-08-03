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
from kivy.uix.screenmanager import ScreenManager, Screen , FadeTransition
import os
#from kivy.uix.screenmanager import FadeTransition
class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.build()

    def build(self):
        layout = FloatLayout()

        self.file_label = Label(text='Drop files here', size_hint=(None, None),
                                pos_hint={'center_x': 0.25, 'center_y': 0.9})
        layout.add_widget(self.file_label)

        next_button = Button(text='Next', size_hint=(None, None), size=(100, 50), pos_hint={'center_x': 0.93, 'center_y': 0.07})
        layout.add_widget(next_button)
        next_button.bind(on_press=self.switch_to_second_screen)
        self.add_widget(layout)


    def switch_to_second_screen(self, instance):
        self.manager.current = 'second'

class SecondScreen(Screen):
    def __init__(self, **kwargs):
        super(SecondScreen, self).__init__(**kwargs)
        self.build()

    def build(self):
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(Label(text='This is the second screen!'))

        back_button = Button(text='Go Back')
        back_button.bind(on_press=self.switch_to_main_screen)
        layout.add_widget(back_button)

        self.add_widget(layout)

    def switch_to_main_screen(self, instance):
        self.manager.current = 'main'

class FileDropApp(App):
    def __init__(self, **kwargs):
        super().__init__()
        self.sm = None

    def build(self):
        self.sm = ScreenManager(transition=FadeTransition())
        self.sm.add_widget(MainScreen(name='main'))
        self.sm.add_widget(SecondScreen(name='second'))

        return self.sm

if __name__ == '__main__':
    FileDropApp().run()

'''
class Graphs_Window(Screen):
    def __init__(self, **kwargs):
        super(Graphs_Window, self).__init__(**kwargs)
        layout = FloatLayout()
        label = Label(text="Welcome to the next screen!",
                      size_hint=(0.6, 0.1), pos_hint={'x': 0.2, 'y': 0.8})
        back_button = Button(text="Back", size_hint=(None, None), size=(100, 50), pos_hint={'x': 0.4, 'y': 0.1})
        back_button.bind(on_press=self.go_back)

        layout.add_widget(label)
        layout.add_widget(back_button)

        self.add_widget(layout)

    def go_back(self, instance):
        self.manager.current = 'main_screen'
        
'''