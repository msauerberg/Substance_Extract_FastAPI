# Substance_Extract_FastAPI
The code creates a API which allows uploading a csv with free text field and returns the extracted substances

# How do run it?

run "uvicorn main:app --reload" in cmd
then run "curl.exe -X 'POST' 'http://127.0.0.1:8000/process/' -F 'file1=@input.csv' -F 'file2=@substanz.csv' -o output.csv" from a second terminal

You can change "input.csv" with another file as long as you keep the column names. I only tested it with semicolon separated csv files.
If you want to use a different reference table, change "substanz.csv" but again, please keep the column names.

# How to run it as a docker container?
Build the container with "docker build -t substance-extract ."
Then, run with "docker run -d -p 8000:8000 substance-extract"
Now, feed your files to the API with "curl.exe -X POST "http://127.0.0.1:8000/process/" -F "file1=@input.csv" -F "file2=@substanz.csv" -o output.csv" from second terminal