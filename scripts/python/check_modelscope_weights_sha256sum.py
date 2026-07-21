from modelscope import HubApi

api = HubApi()
remote_files = api.get_model_files(model_id="Eco-Tech/DeepSeek-V3.2-Exp-w4a8-mtp-QuaRot")
name_sha_remote = {file["Name"]: file["Sha256"] for file in remote_files}

# 在机器上面执行 sha256sum ./quant_* 得到的结果，进行拷贝粘贴此处（因为这个的等待时间很长，没写成在此处实际执行）
local_sha256_name_lines = r"""
2703effcc3d695b64ca807ba6d6d3a4e605631713c825d2c60e44717041bb983  ./quant_model_weights-00083-of-00088.safetensors
fcc74f5b5f4addc6720a390a5549ae0e7d18a500642b56a58268a33a93ea32fc  ./quant_model_weights-00084-of-00088.safetensors
71a79c819aef24ad2e1238c3c82e16a414336f1d55f5f5949df2b54318661ed7  ./quant_model_weights-00085-of-00088.safetensors
e6fd2d7dc113bf3408c1b270953d2f56109b9d20e6f10abef2dd8801de8cf40a  ./quant_model_weights-00086-of-00088.safetensors
"""
name_sha_local = {sp[1][2:]: sp[0] for sp in (line.split() for line in local_sha256_name_lines.strip().splitlines())}

# 比较
diff_flag = False
for name in name_sha_local:
    if name in name_sha_remote:
        if name_sha_local[name] != name_sha_remote[name]:
            print(f"{name}: local sha256sum ({name_sha_local[name]}) != remote sha256sum ({name_sha_remote[name]})")
            diff_flag = True
        else:
            print(f"{name} not found in remote files, please check if the file is missing or renamed.")

if not diff_flag:
    print("All check finish, OK!")