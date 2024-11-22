import scapy.all as scapy
from scapy.all import ARP, Ether, srp
import tensorflow as tf
import numpy as np

def send_arp_request(ip_address):
    arp = ARP(pdst=ip_address)
    ether = Ether(dst="ff:ff:ff:ff:ff:ff")
    packet = ether/arp
    result = srp(packet, timeout=3, verbose=0)[0]
    return result


def train_model(data):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(data[0]),)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(data, labels, epochs=10, batch_size=32)
    return model


def predict(model, data):
    return model.predict(data)

def main():
    ip_addresses = ["192.168.1.1", "192.168.1.2", "192.168.1.3"]  
    data = []
    labels = []

    for ip in ip_addresses:
        result = send_arp_request(ip)
        if result:
            data.append([len(result), len(result[0][1])])  # 
            labels.append(1)  
        else:
            data.append([0, 0]) 
            labels.append(0)  

    data = np.array(data)
    labels = np.array(labels)


    model = train_model(data)

    
    new_data = np.array([[len(send_arp_request("192.168.1.4")), len(send_arp_request("192.168.1.4")[0][1])]])
    prediction = predict(model, new_data)

    if prediction[0][0] > 0.5:
        print("Устройство обнаружено")
    else:
        print("Устройство не обнаружено")

if __name__ == "__main__":
    main()
