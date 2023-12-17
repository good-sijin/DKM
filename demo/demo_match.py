from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from dkm.utils.utils import tensor_to_pil

from dkm import DKMv3_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pickle

def aten_linalg_inv(g, arg):
    return g.op("com.microsoft::Inverse", arg)


# Register custom symbolic function
torch.onnx.register_custom_op_symbolic("aten::linalg_inv", aten_linalg_inv, 17)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)
    parser.add_argument("--save_path", default="demo/dkmv3_warp_sacre_coeur.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    dkm_model = DKMv3_outdoor(device=device)
    
    with open("input.pkl", "rb") as f:
        wc = pickle.load(f)
        query = wc["query"]
        support = wc["support"]

    export=False
    if export:
        # dkm_model.forward = dkm_model.forward_symmetric
        dkm_model.forward = dkm_model.forward_dummy

        input_names=["query", "support"]
        output_names=["output0", "output1"]
        
        with torch.no_grad():
            output = dkm_model(query, support)
            torch.onnx.export(dkm_model, (query, support), "model.onnx", verbose=True,  
                            input_names=input_names, output_names=output_names, opset_version=17)
    else:
        H, W = 864, 1152
        im1 = Image.open(im1_path).resize((W, H))
        im2 = Image.open(im2_path).resize((W, H))
        # # Match
        for i in range(1):
            import time
            st = time.time()
            warp, certainty = dkm_model.match(im1_path, im2_path, device=device)
            et = time.time()
            print(et-st)
        # with open("warp.pkl", "wb") as f:
        #     pickle.dump({"warp": warp, "certainty": certainty}, f)
        
        print(warp.shape, certainty.shape)
        print(warp[0][0], certainty[100][100:103])
        
    # OOM
    exit()
    with open("warp.pkl", "rb") as f:
        wc = pickle.load(f)
        warp = wc["warp"]
        certainty = wc["certainty"]
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    with torch.no_grad():
        dkm_model.sample(warp, certainty)
    x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    im2_transfer_rgb = F.grid_sample(
        x2[None], warp[:, :W, 2:][None], mode="bilinear", align_corners=False
    )[0]
    im1_transfer_rgb = F.grid_sample(
        x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    )[0]
    warp_im = torch.cat((im2_transfer_rgb, im1_transfer_rgb), dim=2)
    white_im = torch.ones((H, 2*W), device=device)
    vis_im = certainty * warp_im + (1 - certainty) * white_im
    tensor_to_pil(vis_im, unnormalize=False).save(save_path)
