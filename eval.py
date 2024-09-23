import os
import csv
import torch

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions


# Running tests
opt = TestOptions().parse(print_options=True)

model_path = opt.model_path
model_name = os.path.basename(model_path).replace('.pth', '')

rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'purity', 'NMI','val loss']]

dataroot = opt.dataroot
vals = os.listdir(opt.dataroot)
print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    print(f"vals:{vals}")
    opt.dataroot = '{}/{}/'.format(dataroot, val)
    print(f"Now opt.dataroot : {opt.dataroot}")
    opt.classes = [''] if 'Real' in os.listdir(opt.dataroot) else os.listdir(opt.dataroot)
    opt.no_resize = True    # testing without resizing by default

    model = resnet50(num_classes=opt.num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, purity, NMI, val_loss, class_report=validate(model, opt)
    rows.append([val, acc, purity, NMI, val_loss])
    print("({}) acc: {}; purity: {}; NMI: {}; val_loss {}".format(val, acc, purity, NMI, val_loss))

# csv_name = os.path.join('./results', f'{model_name}.csv')
# print(f'csv_name: {csv_name}')
# with open(csv_name, 'w') as f:
#     csv_writer = csv.writer(f, delimiter=',')
#     csv_writer.writerows(rows)
