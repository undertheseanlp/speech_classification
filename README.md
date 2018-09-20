# Vietnamese Speech Classification 

Zalo AI Challenge

Repo phục vụ cho Zalo AI Challenge: Voice Gender Classification

Nhóm tác giả:

* Vũ Anh <anhv.ict91@gmail.com>
* Truong Do <truongdq54@gmail.com>
* Nguyễn Thế Anh <nguyentheanhhnvntd@gmail.com>


# Mô tả thuật toán 

*Trích chọn đặc trưng*

Tập tin âm thanh được chia thành những frames dài 250ms không trùng lặp. Mỗi frame được gán với label tương ứng với tập tin âm thanh (gồm gender và ascent). Từ mỗi frame, trích chọn đặc trưng melspectrogram với 60 bands. Kết quả được cho vào hàm logamplitude. 
Tập tin âm thanh đầu vào được đầu bởi thư viện librosa. Sau đó, kết quả với đặc trưng delta tương ứng. 
Kết quả được đưa vào mạng convolutional neural network. 

*Kiến trúc mạng convolutional neural network*

| Layer            | Params                |
|------------------|-----------------------|
| Conv2D           | 64 kernels (7x7)      |
| MaxPooling2D     | size 3x3, strides 2x2 |
| Conv2D           | 128 kernels (5x5)     |
| MaxPooling2D     | size 2x2              |
| Conv2D           | 256 kernels (2x2)     |
| MaxPooling2      | size 2x2              |
| Flatten          |                       |
| Dense            | 200 units             |
| Dropout rate=0.2 |                       |

Mô hình được train bởi hàm tối ưu Adam với learning rate = 0.0001, beta_1 = 0.9, beta=0.999

*Dự đoán*

Mỗi một sample được chia thành mỗi 250 frames, sử dụng phương pháp trích rút đặc trưng như mô tả ở trên, rồi đưa vào mạng CNN. Nhãn của file âm thanh được chọn bởi chiến thuật majority voting.


# Cách sử dụng

## Cài đặt môi trường 

Hệ thống chạy trên môi trường python 3.6, sử dụng keras và tensorflow. Quá trình trích rút đặc trưng được thực hiện bởi thư viện librosa.

```
pip install requirements.txt
```

## Huấn luyện mô hình 

Để huấn luyện mô hình, chạy script `preprocessing.py` và `train.py` 

Chú ý: Dữ liệu train gồm có folder `train` cần đặt vào thư mục `data`

```
python preprocessing.py --mode train
python train.py
```

## Dự đoán 

Để dự đoán, chạy script `predict.py` 

```
python preprocessing.py --mode test
python predict.py 
``` 

