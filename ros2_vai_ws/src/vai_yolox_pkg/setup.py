from setuptools import find_packages, setup

package_name = 'vai_yolox_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alexhegit',
    maintainer_email='heye_dev@163.com',
    description='AMD VAI YOLOX package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "yolox_node = vai_yolox_pkg.yolox_node:main"
        ],
    },
)
