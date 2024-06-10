import sys
import os
import xml.etree.ElementTree as ET  # For SVG conversion

# Add the project root directory and 'pytorch' directory to the Python path
current_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_path)
sys.path.append(root_path)
sys.path.append(current_path)

rendering_path = os.path.join(current_path, 'rendering')
sys.path.append(rendering_path)


from models.model import Model

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
import cv2

from utils import *
from options import parse_args

from datasets.floorplan_dataset import FloorplanDataset
from IP import reconstructFloorplan

from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFileData

# Configure Panda3D to use an offscreen buffer
loadPrcFileData('', 'window-type offscreen')

# Initialize the base ShowBase instance
base = ShowBase()

# Import FloorPlan from floorplan.py
from rendering.floorplan import FloorPlan


# Function to create an SVG element
def create_svg_element(tag, attrib):
    element = ET.Element(tag, attrib)
    return element

# Function to convert custom format to SVG
def convert_to_svg(data):
    # Create the root SVG element
    svg = create_svg_element('svg', {
        'xmlns': "http://www.w3.org/2000/svg",
        'version': "1.1",
        'width': "256",
        'height': "256"
    })

    # Parse the data
    lines = data.strip().split('\n')
    width, height = map(int, lines[0].split())
    num_lines = int(lines[1])
    
    for line in lines[2:2 + num_lines]:
        x1, y1, x2, y2, _1, _2 = map(float, line.split())
        svg_line = create_svg_element('line', {
            'x1': str(x1), 'y1': str(y1),
            'x2': str(x2), 'y2': str(y2),
            'stroke': "black", 'stroke-width': "1"
        })
        svg.append(svg_line)
    
    for line in lines[2 + num_lines:]:
        x1, y1, x2, y2, door_type, _1, _2 = line.split()
        x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
        color = "red" if door_type == "door" else "blue"
        svg_line = create_svg_element('line', {
            'x1': str(x1), 'y1': str(y1),
            'x2': str(x2), 'y2': str(y2),
            'stroke': color, 'stroke-width': "2"
        })
        svg.append(svg_line)

    # Convert the ElementTree to a string
    return ET.tostring(svg, encoding='unicode')

def main(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    print("Checkpoint directory: {}".format(options.checkpoint_dir))
    print("Test directory: {}".format(options.test_dir))

    dataset = FloorplanDataset(options, split='train', random=True)

    print('the number of images', len(dataset))    

    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=True, num_workers=16)

    model = Model(options)

    # Check if CUDA is available and use it; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    if options.restore == 1:
        print('restore')
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth', map_location=device))
        pass
    if options.task == 'test':
        dataset_test = FloorplanDataset(options, split='test', random=False)
        testOneEpoch(options, model, dataset_test, device)
        exit(1)

    optimizer = torch.optim.Adam(model.parameters(), lr = options.LR)
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim.pth', map_location=device))
        pass

    for epoch in range(options.numEpochs):
        epoch_losses = []
        data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
        for sampleIndex, sample in enumerate(data_iterator):
            optimizer.zero_grad()

            images, corner_gt, icon_gt, room_gt = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device)

            corner_pred, icon_pred, room_pred = model(images)
            #print([(v.shape, v.min(), v.max()) for v in [corner_pred, icon_pred, room_pred, corner_gt, icon_gt, room_gt]])
            #exit(1)
            #print(corner_pred.shape, corner_gt.shape)
            #exit(1)
            corner_loss = torch.nn.functional.binary_cross_entropy(corner_pred, corner_gt)
            icon_loss = torch.nn.functional.cross_entropy(icon_pred.view(-1, NUM_ICONS + 2), icon_gt.view(-1))
            room_loss = torch.nn.functional.cross_entropy(room_pred.view(-1, NUM_ROOMS + 2), room_gt.view(-1))
            losses = [corner_loss, icon_loss, room_loss]
            loss = sum(losses)

            loss_values = [l.data.item() for l in losses]
            epoch_losses.append(loss_values)
            status = str(epoch + 1) + ' loss: '
            for l in loss_values:
                status += '%0.5f '%l
                continue
            data_iterator.set_description(status)
            loss.backward()
            optimizer.step()

            if sampleIndex % 500 == 0:
                visualizeBatch(options, images.detach().cpu().numpy(), [('gt', {'corner': corner_gt.detach().cpu().numpy(), 'icon': icon_gt.detach().cpu().numpy(), 'room': room_gt.detach().cpu().numpy()}), ('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 'room': room_pred.max(-1)[1].detach().cpu().numpy()})])
                if options.visualizeMode == 'debug':
                    exit(1)
                    pass
            continue
        print('loss', np.array(epoch_losses).mean(0))
        if True:
            torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint.pth')
            torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim.pth')
            pass

        #testOneEpoch(options, model, dataset_test)        
        continue
    return

def testOneEpoch(options, model, dataset, device):
    model.eval()

    dataloader = DataLoader(dataset, batch_size=options.batchSize, shuffle=False, num_workers=1)

    epoch_losses = []
    data_iterator = tqdm(dataloader, total=len(dataset) // options.batchSize + 1)
    for sampleIndex, sample in enumerate(data_iterator):

        images, corner_gt, icon_gt, room_gt = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device)

        corner_pred, icon_pred, room_pred = model(images)
        corner_loss = torch.nn.functional.binary_cross_entropy(corner_pred, corner_gt)
        icon_loss = torch.nn.functional.cross_entropy(icon_pred.view(-1, NUM_ICONS + 2), icon_gt.view(-1))
        room_loss = torch.nn.functional.cross_entropy(room_pred.view(-1, NUM_ROOMS + 2), room_gt.view(-1))
        losses = [corner_loss, icon_loss, room_loss]
        
        loss = sum(losses)

        loss_values = [l.data.item() for l in losses]
        epoch_losses.append(loss_values)
        status = 'val loss: '
        for l in loss_values:
            status += '%0.5f '%l
            continue
        data_iterator.set_description(status)

        if sampleIndex % 500 == 0:
            print("Saving batch {} to {}".format(sampleIndex, options.test_dir))
            visualizeBatch(options, images.detach().cpu().numpy(), [('gt', {'corner': corner_gt.detach().cpu().numpy(), 'icon': icon_gt.detach().cpu().numpy(), 'room': room_gt.detach().cpu().numpy()}), ('pred', {'corner': corner_pred.max(-1)[1].detach().cpu().numpy(), 'icon': icon_pred.max(-1)[1].detach().cpu().numpy(), 'room': room_pred.max(-1)[1].detach().cpu().numpy()})])
            for batchIndex in range(len(images)):
                corner_heatmaps = corner_pred[batchIndex].detach().cpu().numpy()
                icon_heatmaps = torch.nn.functional.softmax(icon_pred[batchIndex], dim=-1).detach().cpu().numpy()
                room_heatmaps = torch.nn.functional.softmax(room_pred[batchIndex], dim=-1).detach().cpu().numpy()
                print("Reconstructing floorplan for batch {}, image {}".format(sampleIndex, batchIndex))
                output_prefix = options.test_dir + '/' + str(batchIndex) + '_'
                reconstructFloorplan(corner_heatmaps[:, :, :NUM_WALL_CORNERS], corner_heatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4], corner_heatmaps[:, :, -4:], icon_heatmaps, room_heatmaps, output_prefix=output_prefix, densityImage=None, gt_dict=None, gt=False, gap=-1, distanceThreshold=-1, lengthThreshold=-1, debug_prefix='test', heatmapValueThresholdWall=None, heatmapValueThresholdDoor=None, heatmapValueThresholdIcon=None, enableAugmentation=True)
                
                # Convert floorplan to SVG
                floorplan_txt_path = output_prefix + 'floorplan.txt'
                if os.path.exists(floorplan_txt_path):
                    with open(floorplan_txt_path, 'r') as f:
                        floorplan_data = f.read()
                    svg_data = convert_to_svg(floorplan_data)
                    svg_output_path = output_prefix + 'floorplan.svg'
                    with open(svg_output_path, 'w') as f:
                        f.write(svg_data)
                    print(f"Saved SVG to {svg_output_path}")

                # Generate the 3D model
                floorplan = FloorPlan(floorplan_txt_path)
                scene = floorplan.generateEggModel(output_prefix=output_prefix)
                # obj_output_path = output_prefix + 'floorplan.bam'
                # scene.writeBamFile(obj_output_path)
                # print(f"Saved BAM file to {obj_output_path}")


                continue
            if options.visualizeMode == 'debug':
                exit(1)
                pass
        continue
    print('validation loss', np.array(epoch_losses).mean(0))

    model.train()
    return

def visualizeBatch(options, images, dicts, indexOffset=0, prefix=''):
    #cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    #pointColorMap = ColorPalette(20).getColorMap()
    images = ((images.transpose((0, 2, 3, 1)) + 0.5) * 255).astype(np.uint8)
    for batchIndex in range(len(images)):
        image = images[batchIndex].copy()
        filename = options.test_dir + '/' + str(indexOffset + batchIndex) + '_image.png'
        print("Saving image to {}".format(filename))
        cv2.imwrite(filename, image)
        for name, result_dict in dicts:
            for info in ['corner', 'icon', 'room']:
                result_filename = filename.replace('image', '{}_{}'.format(info, name))
                print("Saving {} image for {} to {}".format(info, name, result_filename))
                cv2.imwrite(result_filename, drawSegmentationImage(result_dict[info][batchIndex], blackIndex=0, blackThreshold=0.5))
                continue
            continue
        continue
    return


if __name__ == '__main__':
    args = parse_args()

    args.keyname = 'floorplan'
    #args.keyname += '_' + args.dataset

    if args.suffix != '':
        args.keyname += '_' + suffix
        pass
    
    args.checkpoint_dir = '../checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    print('keyname=%s task=%s started'%(args.keyname, args.task))

    main(args)
