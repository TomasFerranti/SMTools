import os
import time
import json
import cv2 as cv
import copy
import sys
import shutil
import glob

sys.path.append("scripts/")  # nopep8

from shared_functions import readImage, readJson, createImageDict, getStereoFilename, plotCalibSegs
from improve_edges_json import improveJsonEdges
from camera_calibration import calibrateCamera
from stereo_matching import stereoEdgesMatching
from split_image import getStereoSplit


PATHS = {
    "MAIN_FOLDER": "/home/tomas/Documents/FGV/ProjetoRice/ResearchRice-Product2/",
    "TEXTURE_EXTRACTOR": "/home/tomas/Documents/FGV/ProjetoRice/ResearchRice-Product2/TextureExtractor/",
    "STEREO_IMAGES": "imagens_stereo_IMS/",
    "OUTPUT": "processed_data/",
    "CURRENT_IMAGE": "002080BR0206.jpg",
    "CURRENT_CALIB": "002080BR0206_left.json"
}


def saveOutput(filename, data, data_type, output_path="standard"):
    if output_path == "standard":
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


def interfaceBegin(main_title_desc):
    clearScreen()
    print("Welcome to the interface of"+main_title_desc+".\n")
    option_chosen = input("\nShould we start the process? (y/n)\n")
    if option_chosen != "y":
        print("\nOkay, returning to main menu.\n")
        time.sleep(1)
        return False
    return True


def interfaceEnd():
    print("\nProcess complete. All results were saved to",
          PATHS['MAIN_FOLDER'] + PATHS['OUTPUT'], "\n")
    input("\nPress START to continue.\n")


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
    if not interfaceBegin("splitting a stereo image"):
        return

    image_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['STEREO_IMAGES'] + PATHS['CURRENT_IMAGE']
    image_base_name = ''.join(PATHS['CURRENT_IMAGE'].split(sep=".")[:-1])

    try:
        print("Reading image at path", image_input_path, "...", end='')
        img = cv.imread(image_input_path)
        print(" done.")

        print("Splitting the image...", end='')
        imgL, imgR, imgM = getStereoSplit(img)
        print(" done.")

        print("Saving output...", end='')
        saveOutput(image_base_name + "_left.jpg", imgL, "img")
        saveOutput(image_base_name + "_right.jpg", imgR, "img")
        saveOutput(image_base_name + "_middle.jpg", imgM, "img")
        print(" done.")
    except:
        print("\nFailed. Returning to main menu.\n")
        input("\nPress START to continue.\n")
        return

    interfaceEnd()


def improveEdgesInterface():
    if not interfaceBegin("improving calibration edges"):
        return

    image_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT']
    calib_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT'] + PATHS['CURRENT_CALIB']

    try:
        print("Reading calibration at path", calib_input_path, "...", end='')
        img_calib = readJson(calib_input_path)
        print(" done.")

        print("Reading image through calibration data at path",
              image_input_path, "...", end='')
        img = readImage(img_calib, image_input_path)
        print(" done.")

        print("Improving edges...", end='')
        img_calib = improveJsonEdges(img_calib, img)
        print(" done.")

        print("Saving output...", end='')
        saveOutput(PATHS['CURRENT_CALIB'], img_calib, "json")
        print(" done.")
    except:
        print("\nFailed. Returning to main menu.\n")
        input("\nPress START to continue.\n")
        return

    interfaceEnd()


def camCalibInterface():
    if not interfaceBegin("camera calibrating"):
        return

    calib_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT'] + PATHS['CURRENT_CALIB']

    try:
        print("Reading calibration at path", calib_input_path, "...", end='')
        img_calib = readJson(calib_input_path)
        print(" done.")

        print("Calibrating camera...", end='')
        img_calib = calibrateCamera(img_calib)
        print(" done.")

        print("Saving output...", end='')
        saveOutput(PATHS['CURRENT_CALIB'], img_calib, "json")
        print(" done.")
    except:
        print("\nFailed. Returning to main menu.\n")
        input("\nPress START to continue.\n")
        return

    interfaceEnd()


def stereoMatchingInterface():
    if not interfaceBegin("stereo matching"):
        return

    calib_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT'] + PATHS['CURRENT_CALIB']
    image_input_path = PATHS['MAIN_FOLDER'] + \
        PATHS['OUTPUT']

    try:
        print("Reading image1 calibration at path",
              calib_input_path, "...", end='')
        img1_calib = readJson(calib_input_path)
        print(" done.")

        print("Reading image1 through calibration data at path",
              image_input_path, "...", end='')
        img1 = readImage(img1_calib, image_input_path)
        print(" done.")

        print("Creating image1 additional parameters...", end='')
        img1_dict = createImageDict(img1)
        print(" done.")

        print("Creating calibration of image2 from image1...", end='')
        img2_calib = copy.deepcopy(img1_calib)
        img2_calib['nomeImagem'] = getStereoFilename(img1_calib['nomeImagem'])
        print(" done.")

        print("Reading image1 through calibration data at path",
              image_input_path, "...", end='')
        img2 = readImage(img2_calib, image_input_path)
        print(" done.")

        print("Creating image2 additional parameters...", end='')
        img2_dict = createImageDict(img2)
        print(" done.")

        print(
            "Updating calibration of image2 through stereo matching with image1...", end='')
        img2_calib = stereoEdgesMatching(
            img1_calib, img1_dict, img2_calib, img2_dict)
        print(" done.")

        print("Saving output...", end='')
        saveOutput(img2_calib['nomeImagem'] + ".json", img2_calib, "json")
        print(" done.")
    except:
        print("\nFailed. Returning to main menu.\n")
        input("\nPress START to continue.\n")
        return

    interfaceEnd()


def fullPipelineInterface():
    if not interfaceBegin("stereo matching"):
        return

    image_base_path = PATHS['MAIN_FOLDER'] + PATHS['STEREO_IMAGES']
    image_input_path = image_base_path + PATHS['CURRENT_IMAGE']
    image_base_name = ''.join(PATHS['CURRENT_IMAGE'].split(sep=".")[:-1])
    texture_image_path = PATHS['TEXTURE_EXTRACTOR'] + "public/images/"
    texture_calib_path = PATHS['TEXTURE_EXTRACTOR'] + "public/calib/"
    calib_input_path = texture_calib_path + "cab-" + image_base_name + "_left.json"
    output_path = PATHS['MAIN_FOLDER'] + PATHS['OUTPUT']

    try:
        print("Reading image at path", image_input_path, "...", end='')
        img = cv.imread(image_input_path)
        print(" done.")

        print("Splitting the image...", end='')
        imgL, imgR, imgM = getStereoSplit(img)
        print(" done.")

        print("Saving output...", end='')
        saveOutput(image_base_name + "_left.jpg", imgL, "img")
        saveOutput(image_base_name + "_right.jpg", imgR, "img")
        saveOutput(image_base_name + "_middle.jpg", imgM, "img")
        print(" done.")

        print("Copying left image to ", texture_image_path, "...", end='')
        shutil.copy(output_path + image_base_name + "_left.jpg",
                    texture_image_path + image_base_name + "_left.jpg")
        print(" done.")

        input("\nPress enter when calibration is ready at" +
              texture_calib_path + ".\n")

        print("Reading calibration at path", calib_input_path, "...", end='')
        imgL_calib = readJson(calib_input_path)
        print(" done.")

        print("Improving edges of left image...", end='')
        imgL_calib = improveJsonEdges(imgL_calib, imgL)
        print(" done.")

        print("Calibrating camera of left image...", end='')
        imgL_calib = calibrateCamera(imgL_calib)
        print(" done.")

        print("Creating image1 additional parameters...", end='')
        imgL_dict = createImageDict(imgL)
        print(" done.")

        print("Creating calibration of image2 from image1...", end='')
        imgR_calib = copy.deepcopy(imgL_calib)
        imgR_calib['nomeImagem'] = getStereoFilename(imgL_calib['nomeImagem'])
        print(" done.")

        print("Creating image2 additional parameters...", end='')
        imgR_dict = createImageDict(imgR)
        print(" done.")

        print(
            "Updating calibration of image2 through stereo matching with image1...", end='')
        imgR_calib = stereoEdgesMatching(
            imgL_calib, imgL_dict, imgR_calib, imgR_dict)
        print(" done.")

        print("Improving edges of right image...", end='')
        imgR_calib = improveJsonEdges(imgR_calib, imgR)
        print(" done.")

        print("Calibrating camera of right image...", end='')
        imgR_calib = calibrateCamera(imgR_calib)
        print(" done.")

        print("Saving output...", end='')
        saveOutput(PATHS['CURRENT_CALIB'], imgL_calib, "json")
        saveOutput(imgR_calib['nomeImagem'] + ".json", imgR_calib, "json")
        print(" done.")

        plotCalibSegs([imgL_dict, imgR_dict], [imgL_calib, imgR_calib])
    except:
        print("\nFailed. Returning to main menu.\n")
        input("\nPress START to continue.\n")
        return

    interfaceEnd()


def clearFiles():
    paths = [PATHS['MAIN_FOLDER'] + PATHS['OUTPUT'] + "*",
             PATHS['TEXTURE_EXTRACTOR'] + "public/images/*",
             PATHS['TEXTURE_EXTRACTOR'] + "public/calib/*"]
    for path in paths:
        files = glob.glob(path)
        for f in files:
            os.remove(f)


OPTIONS_SCRIPTS = [("0", "Set Environment Variable (MANUAL)", setVariableInterface),
                   ("1", "Split Image (SCRIPT)", splitImageInterface),
                   ("2", "Improve Calibration Edges (SCRIPT)",
                    improveEdgesInterface),
                   ("3", "Calculate Camera Calibration (SCRIPT)", camCalibInterface),
                   ("4", "Find Edges of Stereo Matching (SCRIPT)",
                    stereoMatchingInterface),
                   ("5", "Run Full Pipeline After TextureExtractor (SCRIPT)",
                    fullPipelineInterface),
                   ("6", "Clear All Files Inside Output, TextureExtractor Image and Calibrations Folder (SCRIPT)", clearFiles)]


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
        try:
            OPTIONS_SCRIPTS[int(option_chosen)][2]()
        except:
            print("\nInvalid option.\n")
            time.sleep(1)


def main():
    mainMenu()


if __name__ == "__main__":
    main()
