# EmoVizPhy

## Prerequisites

Before using these commands, Docker must be installed on your system.
If Docker is not yet installed, please follow the instructions provided on the [official Docker website](https://docs.docker.com/get-docker/).

## How to run EmoVizPhy

You must have a text file called `TOKEN.txt` containing
the access token for the GitHub Docker registry.

1. Open your system's terminal.
2. Navigate to the directory where the TOKEN.txt file is located.
3. Run the following command:

   ```shell
   cat ./TOKEN.txt | docker login ghcr.io -u <USERNAME> --password-stdin
   ```

   Where `<USERNAME>` should be replaced with your GitHub username.
   Once the command is executed, you will be successfully authenticated to the remote Docker registry.

4. Run the following command:
   - for Mac computers with Apple Silicon processors (e.g., M1):

     ```shell
     docker pull ghcr.io/collab-uniba/stressawareness:arm64-latest
     ```

   - for Windows computers with 64-bit processors:

     ```shell
     docker pull ghcr.io/collab-uniba/stressawareness:win64-latest
     ```

   This command will download the latest version of the stressawareness image from the GitHub Container Registry to your local system.

5. Run the following command:

   - for Mac computers with Apple Silicon processors (e.g., M1):

     ```shell
     docker run --rm -it -p 20000:20000 ghcr.io/collab-uniba/stressawareness:arm64-latest
     ```

   - for Windows computers with 64-bit processors:

     ```shell
     docker run --rm -it -p 20000:20000 ghcr.io/collab-uniba/stressawareness:win64-latest
     ```

   This command will run the stressawareness image in a container and provide you with the link to use EmoVizPhy in your browser.

## How to contribute

### How to share a new version of the Docker image

To share a new version of the EmoVizPhy Docker image, you need to have a GitHub account
with write permissions on the [collab-uniba GitHub organization](https://github.com/collab-uniba).

1. Build a new version of the Docker image using the labels shown in the following example
   (replace the example tag – i.e., `arm64-latest` – with the desired tag for the new image version):

    ```shell
    docker build \
    --label "org.opencontainers.image.source=https://github.com/collab-uniba/stress_awareness" \
    --label "org.opencontainers.image.description=Visualization tool for biometric data." \
    --pull --rm -f "Dockerfile" -t ghcr.io/collab-uniba/stressawareness:arm64-latest "."
    ```

2. Push the newly created image to the GitHub Container Registry
   (replace the example tag – i.e., `arm64-latest` – with the desired tag for the new image version):

    ```shell
    docker push ghcr.io/collab-uniba/stressawareness:arm64-latest
    ```
