from utils.data import trainloader
from utils.model.Net import *
from config.constants import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for epoch in range(5):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # gradient 0으로 초기화
        optimizer.zero_grad()

        # 순전파, 역전파, 최적화 하기
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계 출력
        running_loss += loss.item()
        if i % 2000 == 1999:    # 2000 mini batch 마다 출력
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# save model
torch.save(net.state_dict(), PATH)
