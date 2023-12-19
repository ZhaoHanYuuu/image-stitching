# 使用SIFT算法实现图像拼接

## 运行方式

本文件夹中包括三个文件，

```python
main.py          # 主函数
mysift.py        # SIFT实现
func.py          # 一些简单的封装函数
```

直接运行`main.py`文件，然后按照提示输入要拼接的图像的序号。因为`images`文件夹中一共包括4组图像，因此可以输入`1,2,3,4`中的任意一个。

结果存储在`result`文件夹中。

## 注意事项

1. 运行前需要先将四组图片拷贝到`images`文件夹中，目前`images`文件夹为**空文件夹**。
2. 运行前需要创建`middle_res`文件夹与`result`文件夹，否则中间结果与最终结果不会保存。
3. 程序运行时间较慢，需要若干小时。
4. 如果想加快运行速度，可以将`main.py`中的36行注释掉，并且取消第38行的注释，即不使用我实现的SIFT，而改为使用`opencv`库中的SIFT，如下所示：

```python
    # 创建SIFT对象并获取关键点和描述子
    # sift = mysift.MySift()
    # 如果调用库函数：
    sift = cv2.SIFT_create()
```

5. 不进行图像拼接，只查看SIFT算法运行结果，可将`main.py`中的第26行进行修改，将`model`的值改为`sift_test`，如下所示：

```python
sift_test = 1
image_stitching = 2
model = sift_test
```

