import os
import torch
# from sgmnet.match_model import matcher as SGM_Modle
from sgmnet.match_model_copy import matcher as SGM_Modle

    
model = SGM_Modle().to("cuda")
model.eval()
# model.load_state_dict(torch.load("/home/freesix/GSN/log/model_best.pth"))
test_data = {
    'x1':torch.rand(1,1000,2).cuda()-0.5,
    'x2':torch.rand(1,1000,2).cuda()-0.5,
    'desc1': torch.rand(1,1000,128).cuda(),
    'desc2': torch.rand(1,1000,128).cuda()
}



trace_model = torch.jit.script(model(test_data))
trace_model.save("model_best.pt")

    

        
# test_data = {
#     'x1':torch.rand(1,1000,2).cuda()-0.5,
#     'x2':torch.rand(1,1000,2).cuda()-0.5,
#     'desc1': torch.rand(1,1000,128).cuda(),
#     'desc2': torch.rand(1,1000,128).cuda()
#     }
# trace_model = torch.jit.script(model, example_inputs=test_data)
# trace_model.save(os.path.join(config.log_base, "model_best.pt"))
