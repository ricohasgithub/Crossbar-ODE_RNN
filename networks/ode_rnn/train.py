
import torch

def train(examples, model, epochs):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.MSELoss()

    training_loss_history = []
    validation_loss_history = []

    # Split data into training and validation
    example_data_length = len(examples)
    val_split_index = example_data_length - int(0.25 * example_data_length)

    for epoch in range(epochs):

        print("EPOCH: ", epoch)

        training_loss = []
        validation_loss = []

        for i, (example, label) in enumerate(examples):

            print(example)
       
            # Validation
            if i > val_split_index:
                with torch.no_grad():
                    prediction = model(example)
                    loss = loss_function(prediction, label)
                    validation_loss.append(loss)
            else:
                # Training
                optimizer.zero_grad()
                prediction = model(example)
                loss = loss_function(prediction, label)
                training_loss.append(loss)
                loss.backward()
                optimizer.step()
    
        # Append avereage loss over batch sample to history
        training_loss_history.append(sum(training_loss) / val_split_index)
        validation_loss_history.append(sum(validation_loss) / (example_data_length - val_split_index))
        
        training_loss = []
        validation_loss = []

    return training_loss_history, validation_loss_history

def test(seq, t, length, model):

    dt = torch.sum(t[1:] - t[0:-1]) / (len(t) - 1)
    output = []
    all_t = []
    
    with torch.no_grad():
        for i in range(length):
            #print(seq)
            #print(seq.size())
            prediction = model((seq, t + dt)).reshape(1, -1, 1)
            seq = torch.cat((seq[1:], prediction), axis=0)
            all_t.append(t[-1].unsqueeze(0) + dt.unsqueeze(0))
            t = torch.cat((t[1:], t[-1].unsqueeze(0) + dt.unsqueeze(0)), axis=0)
            output.append(prediction)

    return torch.cat(output, axis=0), torch.cat(all_t, axis=0)