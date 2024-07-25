<?php
require 'vendor/autoload.php';

use lsolesen\pel\Pel;
use lsolesen\pel\PelConvert;
use lsolesen\pel\PelDataWindow;
use lsolesen\pel\PelEntryAscii;
use lsolesen\pel\PelExif;
use lsolesen\pel\PelIfd;
use lsolesen\pel\PelJpeg;
use lsolesen\pel\PelTag;
use lsolesen\pel\PelTiff;

function println($message) {
    echo $message . "\n";
}

setlocale(LC_ALL, '');

//Function to read image file
function readImageFile($input) {
    ini_set('memory_limit', '32M');
    return new PelDataWindow(file_get_contents($input));
}

//Function to process image metadata
function processImageMetadata($data, &$file) {
    if (PelJpeg::isValid($data)) {
        $jpeg = $file = new PelJpeg();
        $jpeg->load($data);
        $exif = $jpeg->getExif();

        if ($exif == null) {
            $exif = new PelExif();
            $jpeg->setExif($exif);
            $tiff = new PelTiff();
            $exif->setTiff($tiff);
        } else {
            $tiff = $exif->getTiff();
        }
    } elseif (PelTiff::isValid($data)) {
        $tiff = $file = new PelTiff();
        $tiff->load($data);
    } else {
        println('Unrecognized image format! The first 16 bytes follow:');
        PelConvert::bytesToDump($data->getBytes(0, 16));
        exit(1);
    }

    return $tiff;
}

//Function to update image description
function updateImageDescription($tiff, $description) {
    $ifd0 = $tiff->getIfd();

    if ($ifd0 == null) {
        $ifd0 = new PelIfd(PelIfd::IFD0);
        $tiff->setIfd($ifd0);
    }

    $desc = $ifd0->getEntry(PelTag::IMAGE_DESCRIPTION);

    if ($desc == null) {
        $desc = new PelEntryAscii(PelTag::IMAGE_DESCRIPTION, $description);
        $ifd0->addEntry($desc);
    } else {
        $desc->setValue($description);
    }
}

//Function to save modified image
function saveModifiedImage($file, $output) {
    $file->saveFile($output);
}

//Get the data parameters from the web
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $inputImage = $_POST['input_image'] ?? '';
    $outputImage = $_POST['output_image'] ?? '';
    $description = $_POST['description'] ?? '';

    $baseDir = __DIR__; //Base directory where the script is located
    $inputFilePath = $baseDir . '/' . $inputImage;
    $outputFilePath = $baseDir . '/' . $outputImage;

    if (!empty($description)) {
        try {
            $data = readImageFile($inputImage);

            $file = null; //Initialize file variable
            $tiff = processImageMetadata($data, $file);

            updateImageDescription($tiff, $description);

            saveModifiedImage($file, $outputFilePath);

            echo "Image metadata updated and saved successfully!";
        } catch (Exception $e) {
            echo 'Error: ' . $e->getMessage();
        }
    } else {
        echo "Invalid input parameters.";
    }
}
