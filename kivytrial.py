from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
import pandas as pd
from Heart_Failure import predict

class HeartCompanion(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 2  # Adjusted to have two columns for labels and inputs

        # Age input
        self.age_label = Label(text="Enter your age:")
        self.window.add_widget(self.age_label)
        self.userage = TextInput(multiline=False, input_filter="int")
        self.window.add_widget(self.userage)

        # Sex input
        self.sex_label = Label(text="Enter your sex (1 = male, 0 = female):")
        self.window.add_widget(self.sex_label)
        self.usersex = TextInput(multiline=False, input_filter="int")
        self.window.add_widget(self.usersex)

        # Smoke input
        self.smoke_label = Label(text="Do you smoke? (1 = y, 0 = n):")
        self.window.add_widget(self.smoke_label)
        self.usersmoke = TextInput(multiline=False, input_filter="int")
        self.window.add_widget(self.usersmoke)

        # Cholesterol input
        self.chol_label = Label(text="Enter your cholesterol:")
        self.window.add_widget(self.chol_label)
        self.userchol = TextInput(multiline=False, input_filter="int")
        self.window.add_widget(self.userchol)

        # Resting BPS input
        self.bps_label = Label(text="Enter your Resting heartbeats per minute:")
        self.window.add_widget(self.bps_label)
        self.userbps = TextInput(multiline=False, input_filter="int")
        self.window.add_widget(self.userbps)

        # Max Heart Rate Achieved input
        self.max_label = Label(text="Enter your maximum heart rate achieved:")
        self.window.add_widget(self.max_label)
        self.usermax = TextInput(multiline=False, input_filter="int")
        self.window.add_widget(self.usermax)

        # Diabetes input
        self.db_label = Label(text="Do you have diabetes (1 = y, 0 = n):")
        self.window.add_widget(self.db_label)
        self.userdb = TextInput(multiline=False, input_filter="int")
        self.window.add_widget(self.userdb)

        self.button = Button(text="Submit", on_press=self.submit_data)
        self.window.add_widget(self.button)

        return self.window

    def submit_data(self, instance):
        try:
            age = int(self.userage.text)
            sex = int(self.usersex.text)
            smoke = int(self.usersmoke.text)
            chol = int(self.userchol.text)
            bps = int(self.userbps.text)
            max_heart_rate = int(self.usermax.text)
            db = int(self.userdb.text)

            new_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'smoke': [smoke],
                'chol': [chol],
                'trestbps': [bps],
                'thalach': [max_heart_rate],
                'dm': [db]
            })
            predict(new_data)
        except ValueError:
            print("Please enter valid integers for all fields.")

if __name__ == '__main__':
    HeartCompanion().run()
