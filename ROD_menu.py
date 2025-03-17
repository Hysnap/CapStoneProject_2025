
from sl_utils.logger import log_function_call, streamlit_logger


@log_function_call(streamlit_logger)
def pagesetup():
    # Description: This is the main file to set up the menu
    # from app_pages import introduction
    from sl_app_pages.multi_page import MultiPage
    from sl_app_pages.introduction import introduction_body
    from sl_app_pages.notesondataprep import notesondataprep_body
    from sl_app_pages.mod_page_calls import (
         mp1_intro,
         mp2_dataex,
         mp3_datapre,
         mp4_datapre,
         interactive_map,
         loginpage,
         logoutpage,
        #  dcl_combinedcsv,
        #  dcl_fakecsv,
        #  dcl_truecsv,
        )

    # Create an instance of the MultiPage class
    app = MultiPage(app_name="UK Political Donations")  # Create an instance

    # Add your app pages here using .add_page()

    app.add_page("Introduction", introduction_body)
    app.add_page("Objective and Requirements", mp1_intro)
    app.add_page("Data Exploration", mp2_dataex)
    app.add_page("Data Preprocessing", mp3_datapre)
    app.add_page("Data Further Analysis", mp4_datapre)
    # app.add_page("Machine Learning", machinelearning)
    app.add_page("Interactive Map", interactive_map)
    app.add_page("Notes on Data and Manipulations", notesondataprep_body)
    # app.add_page("Fake News Data cleanliness", dcl_fakecsv)
    # app.add_page("True News Data cleanliness", dcl_truecsv)
    # app.add_page("Combined Data cleanliness", dcl_combinedcsv)
    app.add_page("Login", loginpage)
    app.add_page("Logout", logoutpage)

    # app.add_page("Regulated Entities", regulatedentitypage_body)
    app.run()  # Run the  app
    # End of PoliticalPartyAnalysisDashboard.py
# Path: sl_app_pages/ROD_dashboard.py
# end of ROD_dashboard.py
