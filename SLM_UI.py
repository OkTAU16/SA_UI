from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
from kivy.uix.behaviors import DragBehavior
from kivy.core.window import Window
from kivy.graphics import Rectangle, Color, Line, Ellipse
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.checkbox import CheckBox
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.metrics import sp
import os


##### IMAGE POPPING CALSS #####
class ImagePopup(Popup):
    def __init__(self, image_source, **kwargs):
        super().__init__(**kwargs)
        self.title = 'SLM result'
        self.size_hint = (0.8, 0.8)
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(Image(source=image_source))
        self.content = layout


##### COLOR MODIFICATION CLASS #####
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


###### SCREENS SETTINGS ######

class IntroScreen(Screen):
    def __init__(self, **kwargs):
        super(IntroScreen, self).__init__(**kwargs)
        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 1)  # Set the desired color (R, G, B, A)
            self.rect = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_rect, pos=self._update_rect)
        # TODO: change to the git path for the background
        background = Image(source=r"C:\Users\USER\Desktop\לימודים\שנה ד\חישה רציפה\MMR\SLM_logo.png",
                           allow_stretch=True, keep_ratio=False,opacity=0.7)

        # Add the background image to the screen
        self.add_widget(background)
        self.title = None
        self.build()

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def build(self):
        layout = FloatLayout()
        self.title = 'Intro'
        #layout.add_widget(self.title)
        self.title_label = Label(text='[u]SLM Algorithm - Intro Screen[/u]',font_name='times.ttf', bold=True, markup=True,
                                 size_hint=(None, None), font_size='24sp',
                                 pos_hint={'center_x': 0.5, 'top': 0.995}, color=(0.2, 0.2, 0.2, 1))

        layout.add_widget(self.title_label)

        self.instruction_label = Label(text="Welcome to the SLM algorithm! \n"
                                            "The algorithm provides an analysis of trends in a  time series. \n"
                                            "To use the algorithm, please upload your data as [u]'.mat'[/u], [u]'.csv'[/u] or [u]'.xslx'[/u] files.\n"
                                            "Uploading multiple files in a folder is possible as well\n"
                                            "The data should contain the following columns:\n"
                                            "1. [u]Time Series[/u] (optional): time stamp for each sample.\n"
                                            "Not needed for evenly spaced measurements.\n"
                                            "2. [u]Total Energy[/u]: The observable variable of the experiment.\n"
                                            "3. [u]Distance[/u]: distance from final target. At the final target, the distance should be equal to 0.\n"
                                            "A distance column is needed for each target. \n"
                                            "Click on the 'next' button at the bottom right corner to continue.",
                                       font_name='times.ttf', bold=True, size_hint=(None, None),
                                       font_size='18sp', markup=True,
                                       pos_hint={'center_x': 0.5, 'top': 0.65}, color=(0, 0, 0, 1), line_height=sp(0.9))

        layout.add_widget(self.instruction_label)

        next_button = Button(text='Next',font_name='times.ttf', bold=True, size_hint=(None, None), size=(100, 50),
                             pos_hint={'center_x': 0.93, 'center_y': 0.07},background_color=(0.25, 0.41, 0.88, 1))
        layout.add_widget(next_button)
        next_button.bind(on_press=self.switch_to_main_screen)

        self.add_widget(layout)

    def switch_to_main_screen(self, instance):
        self.manager.current = 'main'


class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 1)  # Set the desired color (R, G, B, A)
            self.rect = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_rect, pos=self._update_rect)
        #TODO: change to the git path for the background
        background = Image(source=r"C:\Users\USER\Desktop\לימודים\שנה ד\חישה רציפה\MMR\SLM_logo.png",
                           allow_stretch=True, keep_ratio=False,opacity=0.85)

        # Add the background image to the screen
        self.add_widget(background)
        self.build()

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def build(self):
        Window.bind(on_resize=self.update_drop_area)
        layout = FloatLayout()
        self.layout = layout
        self.title = 'Main'
        # Time series label and buttons
        self.label_time_series = Label(text="Added Time Series", font_name='times.ttf', bold=True, underline=True, size_hint=(None, None),
                                       pos_hint={'center_x': 0.7, 'center_y': 0.9}, color=(0.2, 0.2, 0.2, 1))

        layout.add_widget(self.label_time_series)

        self.button_time_series_yes = ColorCheckBox(group='time_series', size_hint=(None, None), color=(0.13, 0.55, 0.13, 1),
                                                    size=(50, 50), pos_hint={'x': 0.63, 'top': 0.87})

        self.button_time_series_no = ColorCheckBox(group='time_series', size_hint=(None, None), size=(50, 50),
                                                   pos_hint={'x': 0.73, 'top': 0.87})
        layout.add_widget(self.button_time_series_yes)
        layout.add_widget(self.button_time_series_no)

        # PCA labels and buttons
        self.label_PCA = Label(text="PCA", font_name='times.ttf', bold=True, underline=True, size_hint=(None, None),
                               pos_hint={'center_x': 0.7, 'center_y': 0.79}, color=(0.2, 0.2, 0.2, 1))
        layout.add_widget(self.label_PCA)

        self.button_PCA_yes = ColorCheckBox(group='PCA', size_hint=(None, None), color=(0.13, 0.55, 0.13, 1), size=(50, 50),
                                            pos_hint={'x': 0.63, 'top': 0.76})

        self.button_PCA_no = ColorCheckBox(group='PCA', size_hint=(None, None), size=(50, 50),
                                           pos_hint={'x': 0.73, 'top': 0.76})
        layout.add_widget(self.button_PCA_yes)
        layout.add_widget(self.button_PCA_no)
        self.button_PCA_yes.bind(on_press=self.PCA_show)
        self.button_PCA_no.bind(on_press=self.PCA_show)

        self.PCA_input = TextInput(hint_text="Enter number of clusters", font_name='times.ttf', font_size=16, input_filter='int',
                                   multiline=True,
                                   size_hint=(0.1, 0.06), pos_hint={'x': 0.79, 'top': 0.76})
        self.button_PCA_submit = Button(text="✓", size_hint=(None, None), font_name='DejaVuSans.ttf', size=(53, 53),
                                        pos_hint={'x': 0.9, 'top': 0.76}, background_color=(0.25, 0.41, 0.88, 1))
        layout.add_widget(self.PCA_input)
        layout.add_widget(self.button_PCA_submit)
        self.PCA_input.opacity = 0
        self.button_PCA_submit.opacity = 0
        self.button_PCA_submit.bind(on_press=self.PCA_submit)
        self.PCA_input.bind(text=self.text_size_change)

        #Cross validation labels and buttons
        self.label_CV = Label(text="CV", font_name='times.ttf', bold=True, underline=True, size_hint=(None, None),
                              pos_hint={'center_x': 0.7, 'center_y': 0.68}, color=(0.2, 0.2, 0.2, 1))
        layout.add_widget(self.label_CV)

        self.button_CV_yes = ColorCheckBox(group='CV', size_hint=(None, None), color=(0.13, 0.55, 0.13, 1), size=(50, 50),
                                           pos_hint={'x': 0.63, 'top': 0.64})
        self.button_CV_no = ColorCheckBox(group='CV', size_hint=(None, None), size=(50, 50),
                                          pos_hint={'x': 0.73, 'top': 0.64})
        layout.add_widget(self.button_CV_yes)
        layout.add_widget(self.button_CV_no)
        self.button_CV_yes.bind(on_press=self.CV_show)
        self.button_CV_no.bind(on_press=self.CV_show)

        self.CV_input = TextInput(hint_text="Enter number of cross validation",font_name='times.ttf', font_size=16, input_filter='int',
                                  multiline=True,
                                  size_hint=(0.1, 0.06), pos_hint={'x': 0.79, 'top': 0.64})
        self.button_CV_submit = Button(text="✓", size_hint=(None, None), font_name='DejaVuSans.ttf', size=(53, 53),
                                       pos_hint={'x': 0.9, 'top': 0.64}, background_color=(0.25, 0.41, 0.88, 1))
        layout.add_widget(self.CV_input)
        layout.add_widget(self.button_CV_submit)
        self.CV_input.opacity = 0
        self.button_CV_submit.opacity = 0
        self.button_CV_submit.bind(on_press=self.CV_submit)
        self.CV_input.bind(text=self.text_size_change)

        # particle clusters labels and buttons
        self.label_particle_clusters = Label(text="Number of particle Clusters:", font_name='times.ttf', bold=True, underline=True, size_hint=(None, None),
                                             pos_hint={'center_x': 0.67, 'center_y': 0.52}, color=(0.2, 0.2, 0.2, 1))
        layout.add_widget(self.label_particle_clusters)
        self.spinner_particle_clusters = Spinner(
            text='Default',
            values=( '3', '5'),
            size_hint=(0.08, 0.05),
            pos_hint={'x': 0.79, 'top': 0.54},
            font_name='times.ttf',
            background_color=(0.25, 0.41, 0.88, 1)
        )
        layout.add_widget(self.spinner_particle_clusters)
        self.spinner_particle_clusters.bind(text=self.particle_clusters_select)

        # Targets labels and buttons
        self.label_targets = Label(text="Number of targets",font_name='times.ttf', bold=True, underline=True, size_hint=(None, None),
                                   pos_hint={'center_x': 0.7, 'center_y': 0.46}, color=(0.2, 0.2, 0.2, 1))
        layout.add_widget(self.label_targets)
        self.target_input = TextInput(hint_text="Enter number of parameters",font_name='times.ttf', font_size=15, input_filter='int',
                                      multiline=True,
                                      size_hint=(0.09, 0.06), pos_hint={'x': 0.63, 'top': 0.42})
        self.button_targets_submit = Button(text="✓", size_hint=(None, None), font_name='DejaVuSans.ttf', size=(53, 53),
                                            pos_hint={'x': 0.73, 'top': 0.42}, background_color=(0.25, 0.41, 0.88, 1))
        layout.add_widget(self.target_input)
        layout.add_widget(self.button_targets_submit)
        self.target_input.bind(text=self.text_size_change)
        self.button_targets_submit.bind(on_press=self.target_submit)

        self.label_save_path = Label(text="Output directory path",font_name='times.ttf', bold=True, underline=True, size_hint=(None, None),
                                   pos_hint={'center_x': 0.7, 'center_y': 0.32}, color=(0.2, 0.2, 0.2, 1))
        layout.add_widget(self.label_save_path)
        self.save_input = TextInput(hint_text="Enter Output directory path here:", font_name='times.ttf', font_size=22, input_filter=None,
                                      multiline=True,
                                      size_hint=(0.3, 0.06), pos_hint={'x': 0.55, 'top': 0.28})
        self.button_save_submit = Button(text="✓", size_hint=(None, None), font_name='DejaVuSans.ttf', size=(53, 53),
                                            pos_hint={'x': 0.86, 'top': 0.279}, background_color=(0.25, 0.41, 0.88, 1))
        layout.add_widget(self.save_input)
        layout.add_widget(self.button_save_submit)
        self.save_input.bind(text=self.text_size_change_for_path)
        self.button_save_submit.bind(on_press=self.save_path_submit)

        # Create a label to display dropped file path
        self.file_label = Label(text='Drop files here',font_name='times.ttf', bold=True, underline=True, size_hint=(None, None),
                                pos_hint={'center_x': 0.25, 'center_y': 0.9}, color=(0.2, 0.2, 0.2, 1))
        self.file_label.bind(size=self.file_label.setter('text_size'))
        layout.add_widget(self.file_label)
        # Create a test and submit buttons for the process
        test_button = Button(text='Test',font_name='times.ttf', bold=True, size_hint=(None, None), size=(150, 50),
                             pos_hint={'center_x': 0.15, 'center_y': 0.58}, background_color=(0.25, 0.41, 0.88, 1))
        test_button.bind(on_press=self.test_dropped_file)
        self.file_label.bind(size=self.file_label.setter('text_size'))
        layout.add_widget(test_button)

        submit_button = Button(text='Submit', font_name='times.ttf', bold=True, size_hint=(None, None), size=(150, 50),
                               pos_hint={'center_x': 0.35, 'center_y': 0.58}, background_color=(0.25, 0.41, 0.88, 1))
        submit_button.bind(on_press=self.submit_dropped_file)
        layout.add_widget(submit_button)

        # Setting the dropping area
        self.window_width, self.window_height = Window.size
        self.drop_area_x = self.window_width // 16
        self.drop_area_y = 5 * self.window_height // 8
        self.drop_area_width = 3 * self.window_width // 8
        self.drop_area_height = self.window_height // 4
        with layout.canvas:
            Color(0.2, 0.2, 0.2, 1)

            self.line = Line(points=[self.drop_area_x, self.drop_area_y,
                                     self.drop_area_x + self.drop_area_width, self.drop_area_y,
                                     self.drop_area_x + self.drop_area_width, self.drop_area_y + self.drop_area_height,
                                     self.drop_area_x, self.drop_area_y + self.drop_area_height,
                                     self.drop_area_x, self.drop_area_y], width=1)

            #Add a label for testing below the drop area
        self.test_label = Label(text='',font_name='times.ttf', bold=True, size_hint=(None, None),
                                pos_hint={'center_x': 0.25, 'center_y': 0.52}, color=(0.2, 0.2, 0.2, 1))
        layout.add_widget(self.test_label)

        ##### actual buttons of the Main screen #####
        next_button = Button(text='Next', font_name='times.ttf', bold=True, size_hint=(None, None), size=(100, 50),
                             pos_hint={'center_x': 0.93, 'center_y': 0.07}, background_color=(0.25, 0.41, 0.88, 1))
        layout.add_widget(next_button)
        next_button.bind(on_press=self.switch_to_graph_screen)

        back_button = Button(text='back', font_name='times.ttf', bold=True, size_hint=(None, None), size=(100, 50),
                             pos_hint={'center_x': 0.07, 'center_y': 0.07}, background_color=(0.25, 0.41, 0.88, 1))
        layout.add_widget(back_button)
        back_button.bind(on_press=self.switch_to_intro_screen)
        self.add_widget(layout)


        self.instruction_label = Label(text="1. Set model parameters on the right. \n2. Fill in values as necessary and submit, if the value is valid the box will turn green \n3. Drop the file or folder in the drop zone and press the 'test' button. \n4. Submit the data, the results will appear when ready in the next screen.",
                                       font_name='DejaVuSans.ttf', bold=True, size_hint=(None, None),
                                       text_size=(Window.width/2.5, None),markup=True,
                                       pos_hint={'center_x': 0.255, 'top': 0.35}, color=(0.2, 0.2, 0.2, 1))

        layout.add_widget(self.instruction_label)
        self.title_label = Label(text='[u]SLM Algorithm - Main Screen[/u]',font_name='times.ttf', bold=True, markup=True,
                                 size_hint=(None, None), font_size='24sp',
                                 pos_hint={'center_x': 0.5, 'top': 0.995}, color=(0.2, 0.2, 0.2, 1))

        layout.add_widget(self.title_label)
    # Main screen functions
    def switch_to_graph_screen(self, instance):
        self.manager.current = 'graph'

    def switch_to_intro_screen(self, instance):
        self.manager.current = 'intro'

    # Connections function to GuiApp functions
    def PCA_show(self, instance):
        app = App.get_running_app()
        app.PCA_show(instance)

    def PCA_submit(self, instance):
        app = App.get_running_app()
        app.PCA_submit(instance)

    def CV_show(self, instance):
        app = App.get_running_app()
        app.CV_show(instance)

    def CV_submit(self, instance):
        app = App.get_running_app()
        app.CV_submit(instance)

    def particle_clusters_select(self, spinner, text):
        app = App.get_running_app()
        app.particle_clusters_select(spinner, text)

    def target_submit(self, instance):
        app = App.get_running_app()
        app.target_submit(instance)

    def save_path_submit(self, instance):
        app = App.get_running_app()
        app.save_path_submit(instance)

    def text_size_change(self, instance, value):
        app = App.get_running_app()
        app.text_size_change(instance, value)

    def text_size_change_for_path(self, instance, value):
        app = App.get_running_app()
        app.text_size_change_for_path(instance, value)

    def test_dropped_file(self, *args):
        app = App.get_running_app()
        app.test_dropped_file(*args)

    def submit_dropped_file(self, *args):
        app = App.get_running_app()
        app.submit_dropped_file(*args)

    def fill_all_data(self):
        app = App.get_running_app()
        app.fill_all_data(self)

    def update_drop_area(self, Window, *args):
        self.window_width, self.window_height = Window.size
        self.drop_area_x = self.window_width // 16
        self.drop_area_y = 5 * self.window_height // 8
        self.drop_area_width = 3 * self.window_width // 8
        self.drop_area_height = self.window_height // 4
        #self.root.canvas.remove(self.line)
        #self.layout.canvas.clear()
        #with self.root.canvas:
        if self.line:
            self.layout.canvas.remove(self.line)
        with self.layout.canvas:
            #with self.line.canvas:
            Color(0.2, 0.2, 0.2, 1)

            self.line = Line(points=[self.drop_area_x, self.drop_area_y,
                                     self.drop_area_x + self.drop_area_width, self.drop_area_y,
                                     self.drop_area_x + self.drop_area_width, self.drop_area_y + self.drop_area_height,
                                     self.drop_area_x, self.drop_area_y + self.drop_area_height,
                                     self.drop_area_x, self.drop_area_y], width=1)

        if self.instruction_label in self.layout.children:
            self.layout.remove_widget(self.instruction_label)

    # Recreate the instruction label with updated position and size
        self.instruction_label = Label(
        text="1. Set model parameters on the right. \n2. Fill in values as necessary and submit, if the value is valid the box will turn green \n3. Drop the file or folder in the drop zone and press the 'test' button. \n4. Submit the data, the results will appear when ready in the next screen.",
        font_name='DejaVuSans.ttf', bold=True, size_hint=(None, None),
        text_size=(self.window_width / 2.5, None), markup=True,
        pos_hint={'center_x': 0.255, 'top': 0.35}, color=(0.2, 0.2, 0.2, 1))

    # Add the updated instruction label back to the layout
        self.layout.add_widget(self.instruction_label)

class GraphScreen(Screen):
    def __init__(self, **kwargs):
        super(GraphScreen, self).__init__(**kwargs)
        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 1)  # Set the desired color (R, G, B, A)
            self.rect = Rectangle(size=self.size, pos=self.pos)
            self.bind(size=self._update_rect, pos=self._update_rect)
        #TODO: change to the git path for the background
        background = Image(source=r"C:\Users\USER\Desktop\לימודים\שנה ד\חישה רציפה\MMR\SLM_logo.png", allow_stretch=True, keep_ratio=False)

        # Add the background image to the screen
        self.add_widget(background)
        self.build()

    def _update_rect(self, instance, value):
        self.rect.size = instance.size
        self.rect.pos = instance.pos

    def build(self):
        layout = FloatLayout()
        self.title = 'Intro'


        first_graph_button = Button(text="Graph 1",font_name='times.ttf', bold=True, size_hint=(0.3, 0.3), pos_hint={'center_x': 0.3, 'top': 0.65}, background_color=(0.25, 0.41, 0.88, 1))
        second_graph_button = Button(text="Graph 2", font_name='times.ttf', bold=True, size_hint=(0.3, 0.3), pos_hint={'center_x': 0.7, 'top': 0.65}, background_color=(0.25, 0.41, 0.88, 1))
        layout.add_widget(first_graph_button)
        layout.add_widget(second_graph_button)
        first_graph_button.bind(on_release=lambda btn: self.show_image_popup(0))
        second_graph_button.bind(on_release=lambda btn: self.show_image_popup(1))

       # model_button = Button(text="SLM model", font_name='times.ttf', bold=True, size_hint=(0.36, 0.1),
        #                            pos_hint={'center_x': 0.5, 'top': 0.4})
        #layout.add_widget(model_button)

        back_button = Button(text='back', font_name='times.ttf', bold=True, size_hint=(None, None), size=(100, 50),
                             pos_hint={'center_x': 0.07, 'center_y': 0.07})

        # back_button = Button(text='Go Back')
        back_button.bind(on_press=self.switch_to_main_screen)
        layout.add_widget(back_button)

        self.add_widget(layout)

    # Connections function to GuiApp functions

    def show_image_popup(self, graph_index):
        app = App.get_running_app()
        img = os.path.join(app.save_path, app.graph_names[graph_index])
        popup = ImagePopup(image_source=img)
        popup.open()

    # Graphs screen functions
    def switch_to_main_screen(self, instance):
        self.manager.current = 'main'


####### GUI CLASS #####


class GuiApp(App):
    def __init__(self, **kwargs):
        super().__init__()
        self.sm = None
        self.file_path = None
        self.drop_area_height = None
        self.drop_area_width = None
        self.drop_area_y = None
        self.drop_area_x = None
        self.cluster_num = None
        self.CV_num = 3
        self.target_num = None
        self.particle_clusters = 3
        self.save_path = None
        self.graph_names = ['table1.png', 'table1.png']
        #TODO: CHANGE TO REAL GRAPHS NAMES
        self.submit_flag = 0

    def build(self):
        # Allow the window to receive file drops
        Window.bind(on_drop_file=self.on_drop_file)
        #Window.bind(on_resize=self.update_drop_area)
        # Setting the screens
        self.sm = ScreenManager(transition=FadeTransition())
        self.sm.add_widget(IntroScreen(name='intro'))
        self.sm.add_widget(MainScreen(name='main'))
        self.sm.add_widget(GraphScreen(name='graph'))

        return self.sm

    #practicals functions
    def PCA_show(self, instance):
        # Show or hide TextInput and V button based on selected radio button
        main_screen = self.sm.get_screen('main')
        if instance == main_screen.button_PCA_yes and instance.state == 'down':
            main_screen.PCA_input.opacity = 1
            main_screen.button_PCA_submit.opacity = 1
        elif instance == main_screen.button_PCA_no and instance.state == 'down':
            main_screen.PCA_input.opacity = 0
            main_screen.button_PCA_submit.opacity = 0
            self.cluster_num = None
            main_screen.PCA_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            main_screen.PCA_input.background_color = (1, 1, 1, 1)
            main_screen.PCA_input.text = ''
            main_screen.PCA_input.font_size = 16
        else:
            main_screen.PCA_input.opacity = 0
            main_screen.button_PCA_submit.opacity = 0
            self.cluster_num = None
            main_screen.PCA_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            main_screen.PCA_input.background_color = (1, 1, 1, 1)
            main_screen.PCA_input.text = ''
            main_screen.PCA_input.font_size = 16

    def PCA_submit(self, instance):
        # Logging in the PCA input
        main_screen = self.sm.get_screen('main')
        user_input = main_screen.PCA_input.text
        if user_input:
            self.cluster_num = user_input
            print(self.cluster_num)
            main_screen.PCA_input.foreground_color = (0, 0.6, 0, 1)  # RGBA for green
            main_screen.PCA_input.background_color = (0.6, 1, 0.9, 1)
        else:
            self.cluster_num = None
            print(self.cluster_num)
            main_screen.PCA_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color

    def CV_show(self, instance):
        # Show or hide TextInput and V button based on selected radio button
        main_screen = self.sm.get_screen('main')
        if instance == main_screen.button_CV_yes and instance.state == 'down':
            main_screen.CV_input.opacity = 1
            main_screen.button_CV_submit.opacity = 1
        elif instance == main_screen.button_CV_no and instance.state == 'down':
            main_screen.CV_input.opacity = 0
            main_screen.button_CV_submit.opacity = 0
            main_screen.CV_num = None
            main_screen.CV_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            main_screen.CV_input.background_color = (1, 1, 1, 1)
            main_screen.CV_input.text = ''
            main_screen.CV_input.font_size = 16
        else:
            main_screen.CV_input.opacity = 0
            main_screen.button_CV_submit.opacity = 0
            self.CV_num = None
            main_screen.CV_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            main_screen.CV_input.background_color = (1, 1, 1, 1)
            main_screen.CV_input.text = ''
            main_screen.CV_input.font_size = 16

    def CV_submit(self, instance):
        main_screen = self.sm.get_screen('main')
        user_input = main_screen.CV_input.text
        if user_input:
            self.CV_num = user_input
            print(self.CV_num)
            main_screen.CV_input.foreground_color = (0, 0.6, 0, 1)  # RGBA for green
            main_screen.CV_input.background_color = (0.6, 1, 0.9, 1)
        else:
            self.CV_num = None
            print(self.CV_num)
            main_screen.CV_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            main_screen.CV_input.background_color = (1, 1, 1, 1)

    def particle_clusters_select(self, spinner, text):
        main_screen = self.sm.get_screen('main')
        if text == "Default":
            main_screen.particle_clusters = 2
        else:
            main_screen.particle_clusters = int(text)
        print(f"Selected number of clusters: {main_screen.particle_clusters}")

    def target_submit(self, instance):
        main_screen = self.sm.get_screen('main')
        user_input = main_screen.target_input.text
        if user_input:
            self.target_num = user_input
            print(self.target_num)
            main_screen.target_input.foreground_color = (0, 0.6, 0, 1)  # RGBA for green
            main_screen.target_input.background_color = (0.6, 1, 0.9, 1)
        else:
            self.target_num = None
            print(main_screen.target_num)
            main_screen.target_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            main_screen.target_input.background_color = (1, 1, 1, 1)

    def save_path_submit(self, instance):
        main_screen = self.sm.get_screen('main')
        user_input = os.path.abspath(main_screen.save_input.text)
        if user_input:
            if os.path.exists(user_input):
                self.save_path = user_input
                print(self.save_path)
                main_screen.save_input.foreground_color = (0, 0.6, 0, 1)  # RGBA for green
                main_screen.save_input.background_color = (0.6, 1, 0.9, 1)
            else:
                main_screen.test_label.text = "Path provided does not exist"
                main_screen.test_label.color = (1, 0, 0, 1)  # Red color for errors

        else:
            self.save_path = None
            print(main_screen.save_input)
            main_screen.save_input.foreground_color = (0, 0, 0, 1)  # RGBA for default color
            main_screen.save_input.background_color = (1, 1, 1, 1)


    def test_dropped_file(self, *args):
        main_screen = self.sm.get_screen('main')
        # Check if a file is dropped
        if self.file_path is None:
            main_screen.test_label.text = "No file dropped."
            main_screen.test_label.color = (1, 0, 0, 1)  # Red color for errors
            return
        #elif os.path.isdir(self.file_path):

        # Check if user input data is filled correctly
        errors = self.fill_all_data()
        if not errors:
            main_screen.test_label.text = "Please fill all required fields."
            main_screen.test_label.color = (1, 0, 0, 1)  # Red color for errors
            self.submit_flag = 0
            return

        # Check the type of the dropped file or folder
        path = self.file_path
        valid_extensions = ['.mat', '.csv', '.xls', '.xlsx']
        if os.path.isfile(path):
            if not any(path.endswith(ext) for ext in valid_extensions):
                main_screen.test_label.text = "Invalid file type. Only .mat, .csv, and Excel files are allowed."
                main_screen.test_label.color = (1, 0, 0, 1)  # Red color for errors
                self.submit_flag = 0
                return
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if not any(file.endswith(ext) for ext in valid_extensions):
                        main_screen.test_label.text = "Folder contains invalid file types. Only .mat, .csv, and Excel files are allowed."
                        main_screen.test_label.color = (1, 0, 0, 1)  # Red color for errors
                        self.submit_flag = 0
                        return
        else:
            main_screen.test_label.text = "Invalid path. Please drop a file or a folder."
            main_screen.test_label.color = (1, 0, 0, 1)  # Red color for errors
            self.submit_flag = 0
            return

        main_screen.test_label.text = "File or folder validated successfully!"
        main_screen.test_label.color = (0, 0.7, 0, 1)  # Green color for success
        self.submit_flag = 1
        print("File or folder validated successfully!")

    def submit_dropped_file(self, instance):
        main_screen = self.sm.get_screen('main')
        if self.submit_flag == 0:
            main_screen.test_label.text = "Testing is required. \n please fill all required fields and press the test button"
            main_screen.test_label.color = (1, 0, 0, 1)  # Red color for errors
        elif self.submit_flag == 1:
            pass

    def fill_all_data(self):
        """
        making sure that all user data and buttons are pressed
        """
        main_screen = self.sm.get_screen('main')
        errors = []

        # Check Time Series
        if not (main_screen.button_time_series_yes.state == 'down' or main_screen.button_time_series_no.state == 'down'):
            errors.append("Time Series option not selected.")

        # Check PCA
        if not (main_screen.button_PCA_yes.state == 'down' or main_screen.button_PCA_no.state == 'down'):
            errors.append("PCA option not selected.")
        elif main_screen.button_PCA_yes.state == 'down' and self.cluster_num is None:
            errors.append("Number of clusters for PCA not entered.")

        # Check CV
        if not (main_screen.button_CV_yes.state == 'down' or main_screen.button_CV_no.state == 'down'):
            errors.append("CV option not selected.")
        elif main_screen.button_CV_yes.state == 'down' and self.CV_num is None:
            errors.append("Number of ??? for CV not entered.")

        # Check Number of Targets
        if self.target_num is None:
            errors.append("Number of targets not entered.")

        # Check Particle Clusters
        # if not self.spinner_particle_clusters.text or self.spinner_particle_clusters.text == "Default":
        #    errors.append("Number of particle clusters not selected.")

        if errors:
            main_screen.test_label.text = "Errors:\n" + "\n".join(errors)
            main_screen.test_label.color = (1, 0, 0, 1)  # Red color for errors
            print(errors)  # For debugging
            return False
        else:
            main_screen.test_label.text = "All data filled correctly!"
            main_screen.test_label.color = (0, 1, 0, 1)  # Green color for success
            return True



    def text_size_change(self, instance, value, *args):
        print(f"Text changed to: {value}")  # Debug statement
        if value:
            instance.font_size = 28  # Larger font size for user input
            instance.bold = True

        else:
            instance.font_size = 15  # same size as before
            instance.bold = False

    def text_size_change_for_path(self, instance, value, *args):
        print(f"Text changed to: {value}")  # Debug statement
        if value:
            instance.font_size = 14 # Larger font size for user input
            instance.bold = True

        else:
            instance.font_size = 22  # same size as before
            instance.bold = False

    ##### Handeling the dropped file or folder #####
    def on_drop_file(self, window, file_path, x, y):
        # Check if the drop occurred within the specified area
        main_screen = self.sm.get_screen('main')

        if (main_screen.window_width // 16 <= x // (2 / 3) <= main_screen.window_width // 16 + (
                3 * main_screen.window_width // 8)
                and main_screen.window_height // 8 <= y // (2 / 3) <= (3 * main_screen.window_height) // 8):
            print(x)
            print(y)
            # Handle the dropped file
            self.file_path = file_path.decode('utf-8')  # Convert bytes to string
            main_screen.file_label.text = f'Dropped file: {file_path}'
            main_screen.test_label.text = 'File dropped!'
            EXTENTIONS = (".xlsx", ".xlsm", ".xltx", ".xltm")
            print(self.file_path.endswith(EXTENTIONS))
            print(os.path.isfile(file_path))
            # Call a function to process the dropped file
            self.process_dropped_file(file_path)
        else:
            main_screen.file_label.text = 'Drop files only within the specified area!'

    def process_dropped_file(self, file_path):
        # Implement your file processing logic here
        print(f"Processing file: {file_path}")

    # Graph screen functions
    def show_image_popup(self, image_source):
        popup = ImagePopup(image_source)
        popup.open()


if __name__ == '__main__':
    GuiApp().run()


