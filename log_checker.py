import os


if __name__ == "__main__":
    log_filenames = os.listdir("logs")
    for log_filename in log_filenames:
        with open("logs/"+log_filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if "avns_27764" in log_filename:
                print(log_filename)
                print(lines)
                print("-------")