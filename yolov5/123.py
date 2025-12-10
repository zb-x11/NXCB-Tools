import torch

# torch.set_printoptions(precision=23)
# reg = 3.1234567891012345678901234567890
#
# reg_tensor = torch.tensor(reg)
# print("reg  value: ",reg_tensor)
#
# reg_tensor1 = reg_tensor.to(torch.bfloat16)
# print("reg BP16 value: ",reg_tensor1)
#
# reg_tensor2= reg_tensor.to(torch.float16)
# print("reg FP16 value: ",reg_tensor2)
#
# reg_tensor3 = reg_tensor.to(torch.float32)
# print("reg FP32 value: ",reg_tensor3)


torch.set_printoptions(precision=23)
reg1 = 0.0000006
reg1_tensor = torch.tensor(reg1).to(torch.float16)
reg2 = 1.12
reg2_tensor = torch.tensor(reg2).to(torch.float16)
print("FP16 Plus = ",reg1_tensor + reg2_tensor)

reg4_tensor = torch.tensor(reg2).to(torch.float32)
reg3_tensor = torch.tensor(reg1).to(torch.float16)
print("FP32 Plus = ",reg3_tensor + reg4_tensor)

