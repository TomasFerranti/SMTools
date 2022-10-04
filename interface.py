import os
import sys
import time


PATHS = {
    "STEREO_IMAGES_PATH": "imagens_stereo_IMS/",
    "OUTPUT_PATH": "cab_stereo_IMS/",
    "TEXTURE_EXTRACTOR_PATH": "TextureExtractor/",
    "CURRENT_IMAGE": "002080BR0206.jpg",
    "CURRENT_CALIB": "cab-002080BR0206_left.json"
}


def set_variable_interface():
    while True:
        print("\n" * os.get_terminal_size().lines)
        print("Welcome to the interface of setting an enviromnent variable.\n")

        print("Current environment variables:\n")
        for idx, path in enumerate(PATHS.keys()):
            print(idx, path, "(CURRENT VALUE:", PATHS[path], ")")
        print("\n")

        option_chosen = input("\nChoose an option (type Q to exit): \n")
        if option_chosen == "Q":
            break

        value_chosen = input("\nChoose its new value:\n")
        try:
            PATHS[list(PATHS.keys())[int(option_chosen)]] = value_chosen
        except:
            print("\nInvalid option for key or value.\n")


def split_image_interface():
    pass


def improve_edges_interface():
    pass


def cam_calib_interface():
    pass


def stereo_matching_interface():
    pass


def full_pipeline_interface():
    pass


OPTIONS_SCRIPTS = [("0", "Set Environment Variable", set_variable_interface),
                   ("1", "Split Image (SCRIPT)", split_image_interface),
                   ("2", "Improve Calibration Edges (SCRIPT)",
                    improve_edges_interface),
                   ("3", "Calculate Camera Calibration (SCRIPT)", cam_calib_interface),
                   ("4", "Find Edges of Stereo Matching (SCRIPT)",
                    stereo_matching_interface),
                   ("5", "Run Full Pipeline After TextureExtractor (SCRIPT)", full_pipeline_interface)]


def main_menu():
    print("\n" * os.get_terminal_size().lines)
    print("Welcome to the main menu of StereoMatcher. \n \n")

    print("Current environment variables:\n")
    for path in PATHS.keys():
        print(path, " : ", PATHS[path])
    print("\n")

    for option_idx, option_desc, option_func in OPTIONS_SCRIPTS:
        print(option_idx, option_desc)

    option_chosen = input("\nChoose an option (type Q to exit):\n")
    if option_chosen == "Q":
        sys.exit()

    try:
        OPTIONS_SCRIPTS[int(option_chosen)][2]()
    except:
        print("\nInvalid option.\n")
        time.sleep(1)


def main():
    while True:
        main_menu()


if __name__ == "__main__":
    main()
