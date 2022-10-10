import os
import time
import json
import cv2 as cv
import copy

from shared_functions import readImage, readJson, createImageDict, getStereoFilename

from split_image import getStereoSplit
from improve_edges_json import improveJsonEdges
from camera_calibration import calibrateCamera
from stereo_matching import stereoEdgesMatching

PATHS = {
    "MAIN_FOLDER": "/home/tomas/Documents/FGV/ProjetoRice/ResearchRice-Product2/",
    "TEXTURE_EXTRACTOR": "/home/tomas/Documents/FGV/ProjetoRice/ResearchRice-Product2/TextureExtractor/",
    "STEREO_IMAGES": "imagens_stereo_IMS/",
    "OUTPUT": "processed_data/",
    "CURRENT_IMAGE": "002080BR0206.jpg",
    "CURRENT_CALIB": "002080BR0206_left.json"
}


def saveOutput(filename, data, data_type):
    output_path = PATHS['MAIN_FOLDER'] + PATHS['OUTPUT']
    try:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if data_type == "json":
            with open(output_path + filename, 'w') as file:
                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()
            pass
        elif data_type == "img":
            cv.imwrite(output_path + filename, data)
            pass

    except:
        print("\nCouldn't save the output.\nCheck your enviromnent variable for MAIN_FOLDER and OUTPUT\n")


def clearScreen():
    print("\n" * os.get_terminal_size().lines)


def setVariableInterface():
    while True:
        clearScreen()
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


def splitImageInterface():
    image_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['STEREO_IMAGES'] + PATHS['CURRENT_IMAGE']
    image_base_name = ''.join(PATHS['CURRENT_IMAGE'].split(sep=".")[:-1])

    img = cv.imread(image_input_path)

    imgL, imgR, imgM = getStereoSplit(img)

    saveOutput(image_base_name + "_left.jpg", imgL, "img")
    saveOutput(image_base_name + "_right.jpg", imgR, "img")
    saveOutput(image_base_name + "_middle.jpg", imgM, "img")


def improveEdgesInterface():
    image_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT']
    calib_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT'] + PATHS['CURRENT_CALIB']

    img_calib = readJson(calib_input_path)
    img = readImage(img_calib, image_input_path)
    img_calib = improveJsonEdges(img_calib, img)
    saveOutput(PATHS['CURRENT_CALIB'], img_calib, "json")


def camCalibInterface():
    calib_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT'] + PATHS['CURRENT_CALIB']

    data = readJson(calib_input_path)
    data = calibrateCamera(data)
    saveOutput(PATHS['CURRENT_CALIB'], data, "json")


def stereoMatchingInterface():
    calib_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT'] + PATHS['CURRENT_CALIB']
    image_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT']

    img1_calib = readJson(calib_input_path)
    img1 = readImage(img1_calib, image_input_path)
    img1_dict = createImageDict(img1)

    img2_calib = copy.deepcopy(img1_calib)
    img2_calib['nomeImagem'] = getStereoFilename(img1_calib['nomeImagem'])
    img2 = readImage(img2_calib, image_input_path)
    img2_dict = createImageDict(img2)

    img2_calib = stereoEdgesMatching(
        img1_calib, img1_dict, img2_calib, img2_dict)

    saveOutput(img2_calib['nomeImagem'] + "." +
               img2_calib['extensao'], img2_calib, "json")


def fullPipelineInterface():
    pass


OPTIONS_SCRIPTS = [("0", "Set Environment Variable (MANUAL)", setVariableInterface),
                   ("1", "Split Image (SCRIPT)", splitImageInterface),
                   ("2", "Improve Calibration Edges (SCRIPT)",
                    improveEdgesInterface),
                   ("3", "Calculate Camera Calibration (SCRIPT)", camCalibInterface),
                   ("4", "Find Edges of Stereo Matching (SCRIPT)",
                    stereoMatchingInterface),
                   ("5", "Run Full Pipeline After TextureExtractor (SCRIPT)", fullPipelineInterface)]


def mainMenu():
    while True:
        clearScreen()
        print("Welcome to the main menu of StereoMatcher. \n \n")

        print("Current environment variables:\n")
        for path in PATHS.keys():
            print(path, " : ", PATHS[path])
        print("\n")

        for option_idx, option_desc, option_func in OPTIONS_SCRIPTS:
            print(option_idx, option_desc)

        option_chosen = input("\nChoose an option (type Q to exit):\n")
        if option_chosen == "Q":
            break

        OPTIONS_SCRIPTS[int(option_chosen)][2]()
        # try:
        #    OPTIONS_SCRIPTS[int(option_chosen)][2]()
        # except:
        #    print("\nInvalid option.\n")
        #    time.sleep(1)


def main():
    mainMenu()


if __name__ == "__main__":
    main()
