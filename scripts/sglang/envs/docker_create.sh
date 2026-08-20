# 有什么用？
# 根据输入的镜像创建NPU容器

# 如何使用？
# 1. 一般修改容器名称，以及最后一行镜像名称就行
# 2. 再另外按照自己的需求修改脚本
# 3. 运行脚本 bash xx.sh

# A3 设备脚本
docker run -itd --name sgl-xjw-c9 --shm-size=16g \
    --privileged=true --net=host \
    -v /mnt:/mnt -v /home:/home -v /data:/data \
    -v /var/queue_schedule:/var/queue_schedule \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    --device=/dev/davinci0:/dev/davinci0 \
    --device=/dev/davinci1:/dev/davinci1 \
    --device=/dev/davinci2:/dev/davinci2 \
    --device=/dev/davinci3:/dev/davinci3 \
    --device=/dev/davinci4:/dev/davinci4 \
    --device=/dev/davinci5:/dev/davinci5 \
    --device=/dev/davinci6:/dev/davinci6 \
    --device=/dev/davinci7:/dev/davinci7 \
    --device=/dev/davinci8:/dev/davinci8 \
    --device=/dev/davinci9:/dev/davinci9 \
    --device=/dev/davinci10:/dev/davinci10 \
    --device=/dev/davinci11:/dev/davinci11 \
    --device=/dev/davinci12:/dev/davinci12 \
    --device=/dev/davinci13:/dev/davinci13 \
    --device=/dev/davinci14:/dev/davinci14 \
    --device=/dev/davinci15:/dev/davinci15 \
    --device=/dev/davinci_manager:/dev/davinci_manager \
    --device=/dev/hisi_hdc:/dev/hisi_hdc \
    --entrypoint=bash \
    swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.0.0-a3-20260602

exit 0

# A2 设备脚本
# todo


# A5 设备脚本
# todo
