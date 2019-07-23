
# Exclude from all to prevent installs in dependencies
add_subdirectory("murmur-cuda/src" EXCLUDE_FROM_ALL)

set(depend_src dependencies/murmur-cuda/src PARENT_SCOPE)
