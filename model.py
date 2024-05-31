import torch
import torch.nn as nn
import torchvision.transforms.functional as tf


def convolution_block(in_channels: int, out_channels: int):
    """Creates and returns a CNN block for the encoders and decoders

    Parameters
    ----------
    in_channels : int
        The number of input channels received by the block.

    out_channels : int
        The number of output channels the block should generate.

    Returns
    -------
    nn.Sequential
        Returns an nn.Sequential module
    """

    CNN_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    return CNN_block


class CNNBlock(nn.Module):

    def __init__(self, n_layers, in_channels, out_channels):
        """Creates a CNN block which acts as a building block
        for the encoder and decoders!

        Parameters
        ----------
        n_layers : int
            The total number of layers of CNNs

        in_channels : int
            The number of input channels

        out_channels : int
            The number of output channels
        """
        super().__init__()

        # Defining our block!
        self.block = nn.ModuleList()

        for _ in range(n_layers):

            self.block.append(convolution_block(in_channels, out_channels))

            # After the first layer, in_channels should match out channels
            in_channels = out_channels

    def forward(self, X):

        # Pass the input through each layer
        for layer in self.block:
            X = layer(X)

        return X


class Encoder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, layers: int):
        """Creates the Encoder block for our UNet

        Parameters
        ----------
        in_channels : int
            The input channels the Encoder accepts

        out_channels : int
            The output channels the Encoder generates

        layers : int
            The total layers that will form the Encoder
        """
        super().__init__()

        # Defining our encoder!
        self.encoder = nn.ModuleList()

        for _ in range(layers):

            # Adding in a CNN layer and maxpooling to reduce 2D dimensions
            CNN_block = CNNBlock(2, in_channels, out_channels)
            self.encoder += [CNN_block, nn.MaxPool2d(2, 2)]

            # This will eventually match out channels
            in_channels = out_channels
            out_channels *= 2

        # Add in the final layer (the information bridge from the diagram!)
        self.encoder.append(
            CNNBlock(
                n_layers=2,
                in_channels=in_channels,
                out_channels=out_channels,
            )
        )

    def forward(self, X):
        """Returns processed X and the skip connection inputs

        Returns
        -------
        Tuple
            A tuple containing the processed X value (0th index)
            and the skip connection inputs (1st index)
        """

        # The outputs from each layer will be passed to the decoder
        skip_connection_input = []

        for layer in self.encoder:

            X = layer(X)

            # Add the outputs of the CNNBlocks to the skip connection input!
            if isinstance(layer, CNNBlock):
                skip_connection_input.append(X)

        # Return the value
        return (X, skip_connection_input)


class Decoder(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exit_channels: int,
        layers: int,
    ):
        """Creates the Decoder block for our UNet

        Parameters
        ----------
        in_channels : int
            The input channels the Decoder accepts

        out_channels : int
            The output channels the Decoder generates

        exit_channels : int
          The final output channels of the UNet (equals total classes)

        layers : int
            The total layers that will form the Decoder
        """
        super().__init__()

        # Defining our encoder!
        self.decoder = nn.ModuleList()

        for _ in range(layers):

            # Adding in a CNN layer and maxpooling
            CNN_block = CNNBlock(2, in_channels, out_channels)

            self.decoder += [
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
                CNN_block,
            ]

            in_channels //= 2
            out_channels //= 2

        self.decoder.append(
            nn.Conv2d(in_channels, exit_channels, kernel_size=1, padding=0)
        )

    def forward(self, X, skip_connections):
        """Processes X and the skip connection inputs

        Returns
        -------
        torch.Tensor
            Returns the processed X
        """

        # Remove the last connection since it is attached
        # to the Information Bridge
        skip_connections.pop(-1)

        for layer in self.decoder:

            # Do not do this for the Transpose Convolution layers
            if isinstance(layer, CNNBlock):

                # Add the skip connection value to our data!
                skip_connections[-1] = tf.center_crop(
                    skip_connections[-1], X.shape[2]
                )

                X = torch.cat([X, skip_connections.pop(-1)], dim=1)

            # Finally, process the input
            X = layer(X)

        return X


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, class_count, layers):

        super().__init__()

        # Defining our encoder block
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=out_channels,
            layers=layers,
        )

        # Explanation for: in_channels = out_channels * (2**layers)
        # The out_channels value represents the output value from
        # the first CNNBlock. At the end of the layers, it will be
        # incremented by 2**(amount of layers!)

        # Defining our decoder block
        self.decoder = Decoder(
            out_channels * (2**layers),
            out_channels * (2 ** (layers - 1)),
            class_count,
            layers,
        )

    def forward(self, X):

        X, skip_connections = self.encoder(X)
        X = self.decoder(X, skip_connections)

        return X
