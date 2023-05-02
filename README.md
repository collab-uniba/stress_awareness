## How to download and use EmoVizPhy


### Prerequisites


Before using these commands, Docker must be installed on your system. If Docker is not yet installed, please follow the instructions provided on the official Docker website: https://docs.docker.com/get-docker/


### Instructions for download EmoVizPhy
you must have a text file called `TOKEN.txt` containing 
the access token for the GitHub Docker registry. 

1. Open your system's terminal.
2. Navigate to the directory where the TOKEN.txt file is located.
3. Run the following command:

`cat ./TOKEN.txt | docker login ghcr.io -u <USERNAME> --password-stdin`

Where `<USERNAME>` should be replaced with your GitHub username.
Once the command is executed, you will be successfully authenticated to the remote Docker registry.

4. Run the following command:

`docker pull ghcr.io/collab-uniba/stressawareness:arm64-latest` (for mac m1)
`docker pull ghcr.io/collab-uniba/stressawareness:win64-latest` (for windows 64 bit)

This command will download the latest version of the stressawareness image from the GitHub Container Registry to your local system.

5. Run the following command:

`docker run --rm -it -p 20000:20000 ghcr.io/collab-uniba/stressawareness:arm64-latest` (for mac m1)
`docker run --rm -it -p 20000:20000 ghcr.io/collab-uniba/stressawareness:arm64-latest` (for windows 64 bit)


This command will run the stressawareness image in a container and provide you with the link to use EmoVizPhy in your browser.

