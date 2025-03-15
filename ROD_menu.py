

from sl_utils.logger import log_function_call


@log_function_call
def pagesetup():
    # Description: This is the main file to set up the menu
    # from app_pages import introduction
    from sl_app_pages.multi_page import MultiPage
    from sl_app_pages.introduction import introduction_body
    # from app_pages.headlinefigures import hlf_body
    from sl_app_pages.notesondataprep import notesondataprep_body
    from sl_app_pages.mod_page_calls import (
         loginpage,
         logoutpage,)

    # Create an instance of the MultiPage class
    app = MultiPage(app_name="UK Political Donations")  # Create an instance

    # Add your app pages here using .add_page()
    app.add_page("Introduction", introduction_body)
    app.add_page("Login", loginpage)
    app.add_page("Notes on Data and Manipulations", notesondataprep_body)
    app.add_page("Logout", logoutpage)

    # app.add_page("Regulated Entities", regulatedentitypage_body)
    app.run()  # Run the  app
    # End of PoliticalPartyAnalysisDashboard.py
