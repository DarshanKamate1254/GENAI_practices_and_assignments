import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class GitAutoCommit(FileSystemEventHandler):

    def on_modified(self, event):
        if event.is_directory:
            return

        try:
            print("File changed:", event.src_path)

            subprocess.run(["git", "add", "."])
            subprocess.run(["git", "commit", "-m", "auto update"])
            subprocess.run(["git", "push"])

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    path = "."
    event_handler = GitAutoCommit()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()

    print("Auto Git Commit Running...")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()