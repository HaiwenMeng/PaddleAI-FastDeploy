基于fastDeploy的新版paddlex-beta模型C++推理部署

简介：

近期百度开放了paddlex-beta内测，与旧版paddlex的模型推理不兼容，遂新建此仓库，基于paddle的fastDeploy，使用QT/C++重新梳理了一下paddlex模型的推理部署。实现包括class、detect、segment三种模型类别。



编译环境：

QT 5.15

fastdeploy-win-x64

msvc 2019-64bit

opencv4.6



一、编译fastdeploy库

官方提供的dll是opencv3.4.16编的，如果opencv版本一致，可直接下载官方提供的fastdeploy-win-x64-1.0.7 ([C++SDK Release版本](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/download_prebuilt_libraries.md))，不用编译。

笔者此处使用opencv4.6重新编译：
需要注意的是，此库只支持release编译，CMAKE_CONFIG_TYPES应只选择release。
然后OPENCV_DIRECTORY更换至opencv460路径即可。

注意：如果用自编译的dll进行lib二次开发，需要用release，即.pro文件中添加:

```cpp
QMAKE_CXXFLAGS_RELEASE += -MT
QMAKE_CFLAGS_DEBUG += -MT

/*
多线程调试Dll (/MDd) 对应的是MD_DynamicDebug
多线程Dll (/MD) 对应的是MD_DynamicRelease
多线程(/MT) 对应的是MD_StaticRelease
多线程(/MTd)对应的是MD_StaticDebug
*/
```

否则会报错：error LNK2038: 检测到“_ITERATOR_DEBUG_LEVEL”的不匹配项



二、推理注意事项

1>数据类型，全部继承至```fastdeploy::FastDeployModel```类：

![1694397561152](readme.assets/1694397561152.png)

2>新版模型文件的config.yml不提供标签，需要自己提供labels文件，根据结果的index自行解析label。

3> fastdeploy并不是所有的det模型都做nms后处理，需要自己调用cv::dnn:nms等接口实现。
