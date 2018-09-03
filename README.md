# Zalo AI Challenge

Repo phục vụ cho Zalo AI Challenge: Voice Gender Classification

Thời gian: 25/08/2018

Nhóm tác giả:

* Vũ Anh <anhv.ict91@gmail.com>
* Truong Do <truongdq54@gmail.com>


# Mô tả thuật toán 

*Trích chọn đặc trưng*

Tập tin âm thanh được chia thành những frames dài 250ms không trùng lặp. Mỗi frame được gán với label tương ứng với tập tin âm thanh (gồm gender và ascent). Từ mỗi frame, trích chọn đặc trưng melspectrogram với 60 bands. Kết quả được cho vào hàm logamplitude. 
Tập tin âm thanh đầu vào được đầu bởi thư viện librosa. Sau đó, kết quả với đặc trưng delta tương ứng. 
Kết quả được đưa vào mạng convolutional neural network. 

*Kiến trúc mạng convolutional nerual network*

Layer (type)                 Params                    Param #   
=================================================================
Conv2D                       64 kernels (7x7)          12672     
_________________________________________________________________
MaxPooling2D                 size 3x3, strides 2x2         
_________________________________________________________________
Conv2D                       128 kernels (5x5)         12672         
_________________________________________________________________
MaxPooling2D                 size 2x2         
_________________________________________________________________
Conv2D                       256 kernels (2x2)         590080    
_________________________________________________________________
MaxPooling2                  size 2x2                 
_________________________________________________________________
Flatten         
_________________________________________________________________
Dense                        200 units               358600    
_________________________________________________________________
Dropout                      rate=0.2         
_________________________________________________________________
Dense                        softmax      
=================================================================

Mô hình được train bởi hàm tối ưu Adam với learning rate = 0.0001, beta_1 = 0.9, beta=0.999

*Dự đoán*

Mỗi một sample được chia thành mỗi 250 frames, sử dụng phương pháp trích rút đặc trưng như mô tả ở trên, rồi đưa vào mạng CNN. Nhãn của file âm thanh được chọn bởi phương pháp Majority voting.


# Cách sử dụng

## Cài đặt môi trường 

Hệ thống chạy trên môi trường python 3.6, sử dụng keras và tensorflow. Quá trình trích rút đặc trưng được thực hiện bởi thư viện librosa.

```
pip install requirements.txt
```

## Huấn luyện mô hình 

Để huấn luyện mô hình, chạy script `make_sample.py` và `train.py` 

```
python make_sample.py
python train.py
```

## Dự đoán 

Để dự đoán, chạy script `predict.py` 

```
python predict.py 
``` 

