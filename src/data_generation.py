import slicer 
import numpy as np
import os
import vtk
import sys

# run: python data_generation.py "without_contrast_volume.nii" "with_contrast_volume.nii" "vessel_map_volume.nii" "/path/to/store/data" 100

# sanity check
assert len(sys.argv) == 6, 'Additionnal information is required'
for i in range(1, 4):
    assert os.path.isfile(os.path.isfile(sys.argv[i])), f"No such a file: {sys.argv[i]}"
assert os.path.exists(sys.argv[4]), f"No such a directory: {sys.argv[4]}"
assert isinstance(sys.argv[5], int), 'Number of samples should be an integer'


slicer.mrmlScene.Clear()

# CT without contrast volume
volumeNode = slicer.util.loadVolume(sys.argv[1])

# CT with contrast volume
contrastNode = slicer.util.loadVolume(sys.argv[2])

# Vessel Map volume
stcNode = slicer.util.loadVolume(sys.argv[3])


outputVolumeFilenamePattern = sys.argv[4] +  "/%04d/deformed_no_contrast.nii"
outputVolumeFilenamePattern_stc = sys.argv[4] + "/%04d/deformed_vm.nii"
outputVolumeFilenamePattern_contrast = sys.argv[4] + "/%04d/deformed_with_contrast.nii"


# Number of samples to generate
numberOfOutputVolumesToCreate = 250

# Transformation parameters
translationStDev = 0
rotationDegStDev = 0
warpingControlPointsSpacing = 20
warpingDisplacementStdDev = 5

# Create output folders
for filepath in [outputVolumeFilenamePattern]:
    filedir = os.path.dirname(filepath)
    if not os.path.exists(filedir):
        os.makedirs(filedir)

# Set up warping transform computation
pointsFrom = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PointsFrom")
pointsFrom.SetLocked(True)
pointsFrom.GetDisplayNode().SetPointLabelsVisibility(False)
pointsFrom.GetDisplayNode().SetSelectedColor(0,1,0)
pointsTo = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "PointsTo")
pointsTo.GetDisplayNode().SetPointLabelsVisibility(False)
volumeBounds=[0,0,0,0,0,0]
volumeNode.GetBounds(volumeBounds)
warpingTransformNode = None

if hasattr(slicer.modules, "fiducialregistrationwizard"):
    warpingTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "WarpingTransform")
    fidReg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLFiducialRegistrationWizardNode")
    fidReg.SetRegistrationModeToWarping()
    fidReg.SetAndObserveFromFiducialListNodeId(pointsFrom.GetID())
    fidReg.SetAndObserveToFiducialListNodeId(pointsTo.GetID())
    fidReg.SetOutputTransformNodeId(warpingTransformNode.GetID())
else:    
    slicer.util.errorDisplay("SlicerIGT extension is required for applying warping transform")

# Set up linear transform computation
fullTransformNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode", "FullTransform")
fullTransformNode.SetAndObserveMatrixTransformToParent(vtk.vtkMatrix4x4())
volumeNode.SetAndObserveTransformNodeID(fullTransformNode.GetID())
stcNode.SetAndObserveTransformNodeID(fullTransformNode.GetID())
contrastNode.SetAndObserveTransformNodeID(fullTransformNode.GetID())

# Set up transformation chain: volume is warped, then translated&rotated
transformedVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
transformedstcNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
transformedcontrastNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")

parameters = {
    "inputVolume": volumeNode.GetID(),
    "outputVolume": transformedVolumeNode.GetID(),
    "referenceVolume": volumeNode.GetID(),
    "transformationFile": fullTransformNode.GetID()}
    
stc_parameters = {
    "inputVolume": stcNode.GetID(),
    "outputVolume": transformedstcNode.GetID(),
    "referenceVolume": stcNode.GetID(),
    "transformationFile": fullTransformNode.GetID()}
    
contrast_parameters = {
    "inputVolume": contrastNode.GetID(),
    "outputVolume": transformedcontrastNode.GetID(),
    "referenceVolume": contrastNode.GetID(),
    "transformationFile": fullTransformNode.GetID()}

# Initial resampling (without transformation)
resampleParameterNode = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, parameters)
resampleParameterStcNode = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, stc_parameters)
resampleParametercontrastNode = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, None, stc_parameters)

# Set up visualization for screenshots
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
slicer.util.setSliceViewerLayers(background=transformedVolumeNode, fit=True)
pointsFrom.GetDisplayNode().SetVisibility(False)
pointsTo.GetDisplayNode().SetVisibility(False)
slicer.app.layoutManager().threeDWidget(0).mrmlViewNode().SetBackgroundColor(0,0,0)
slicer.app.layoutManager().threeDWidget(0).mrmlViewNode().SetBackgroundColor2(0,0,0)

# Volume rendering
volRenLogic = slicer.modules.volumerendering.logic()
displayNode = volRenLogic.CreateDefaultVolumeRenderingNodes(transformedVolumeNode)
displayNode.SetVisibility(True)
scalarRange = transformedVolumeNode.GetImageData().GetScalarRange()
if scalarRange[1]-scalarRange[0] < 1500:
    # small dynamic range, probably MRI
    displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName('MR-Default'))
else:
    # larger dynamic range, probably CT
    displayNode.GetVolumePropertyNode().Copy(volRenLogic.GetPresetByName('CT-Chest-Contrast-Enhanced'))

# Generate as many deformed volumes as requested
for outputVolumeIndex in range(numberOfOutputVolumesToCreate):
    # translation and rotation
    
    fullTransform = vtk.vtkGeneralTransform()
    if warpingTransformNode:
    # warping
        controlPointCoordsSplit = np.mgrid[
            volumeBounds[0]:volumeBounds[1]:warpingControlPointsSpacing,
            volumeBounds[2]:volumeBounds[3]:warpingControlPointsSpacing,
            volumeBounds[4]:volumeBounds[5]:warpingControlPointsSpacing]
        controlPointCoords = np.vstack([controlPointCoordsSplit[0].ravel(), controlPointCoordsSplit[1].ravel(), controlPointCoordsSplit[2].ravel()]).T
        controlPointDisplacement = np.random.normal(0, warpingDisplacementStdDev, size=controlPointCoords.shape)
        slicer.util.updateMarkupsControlPointsFromArray(pointsFrom, controlPointCoords)
        slicer.util.updateMarkupsControlPointsFromArray(pointsTo, controlPointCoords + controlPointDisplacement)
        fullTransform.Concatenate(warpingTransformNode.GetTransformFromParent())
    fullTransform.Translate(np.random.normal(0, translationStDev, 3))
    fullTransform.RotateX(np.random.normal(0, rotationDegStDev))
    fullTransform.RotateY(np.random.normal(0, rotationDegStDev))
    fullTransform.RotateZ(np.random.normal(0, rotationDegStDev))
    fullTransformNode.SetAndObserveTransformFromParent(fullTransform)
    
    # Save No_contrast volume
    parameters["inputVolume"] = volumeNode.GetID()
    parameters["interpolationType"] = "nn"
    resampleParameterNode = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, resampleParameterNode, parameters)
    
    # Save STC volume
    stc_parameters["inputVolume"] = stcNode.GetID()
    stc_parameters["interpolationType"] = "nn"
    resampleParameterStcNode = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, resampleParameterStcNode, stc_parameters)
    
    # Save Contrast Volume
    contrast_parameters["inputVolume"] = contrastNode.GetID()
    contrast_parameters["interpolationType"] = "nn"
    resampleParametercontrastNode = slicer.cli.runSync(slicer.modules.resamplescalarvectordwivolume, resampleParametercontrastNode, contrast_parameters)
    
    # Save result volume
    outputFilename = outputVolumeFilenamePattern % outputVolumeIndex
    outputFilename_stc = outputVolumeFilenamePattern_stc % outputVolumeIndex
    outputFilename_contrast = outputVolumeFilenamePattern_contrast % outputVolumeIndex
    print("Save transformed volume {0}/{1} as {2}".format(outputVolumeIndex+1, numberOfOutputVolumesToCreate, outputFilename))
    
    success = slicer.util.saveNode(transformedVolumeNode, outputFilename)
    success_stc = slicer.util.saveNode(transformedstcNode, outputFilename_stc)
    success_contrast = slicer.util.saveNode(transformedcontrastNode, outputFilename_contrast)
    