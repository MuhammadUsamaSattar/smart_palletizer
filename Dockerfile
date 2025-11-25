# syntax=docker/dockerfile:1
ARG ROS_DISTRO=jazzy
FROM ros:${ROS_DISTRO} AS base

# ------------------- 1) System deps -------------------
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-colcon-common-extensions python3-rosdep \
        build-essential git cmake wget curl ca-certificates \
        locales dbus-user-session tini \
    && rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ------------------- 2) Workspace -------------------
ARG WS=/workspace
WORKDIR ${WS}

# Copy source
COPY ./src ./src

# ------------------- 3) Build workspace -------------------
RUN rosdep update || true
RUN apt-get update && rosdep install --from-paths src -yi
SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --install-base /workspace/install

# ------------------- 4) Runtime image -------------------
FROM ros:${ROS_DISTRO}

ARG WS=/workspace
WORKDIR ${WS}

# Runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip dbus-user-session locales tini \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Copy workspace from builder
COPY --from=base ${WS}/install ./install
COPY --from=base ${WS}/src ./src

# Source ROS and workspace by default
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc && \
    echo "source ${WS}/install/setup.bash" >> ~/.bashrc

# Default entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
