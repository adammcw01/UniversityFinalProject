from flask import Flask, render_template, request, flash, redirect
import Utilities
import Main

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    This function renders the home page and handles http requests.
    If any errors occur this function will display them in an alert box through flash
    :return: A http redirect to the home page
    """
    if request.method == 'POST':
        if 'AudioFile' not in request.files:
            flash('Form could not find file')
            return redirect(request.url)
        file = request.files['AudioFile']
        tags = request.form
        if not Utilities.checkFileType(file.filename):
            flash('File is invalid format')
            return redirect(request.url)
        filepath = Utilities.getFilePath(file)
        file.save(filepath)
        if not Utilities.tagMetadata(filepath, tags):
            flash('File could not be encoded properly')
            return redirect(request.url)
    """
    Main.main() is responsible for creating an instance of the backend application
    """
    result = Main.main()
    if result:
        flash(result)
    return render_template('index.html')

if __name__ == '__main__':
    """
    This is the entry point for the web server.
    A Flask app is configured to run on port 8000
    A secret key is setup to allow POST requests to the server.
    """
    Utilities.setupFilesystem()
    app.secret_key = '8B94BFE66DE11E295D925B73A4DD6'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run(debug=True, port=8000)
