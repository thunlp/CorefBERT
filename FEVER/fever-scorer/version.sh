export VERSION_NUMBER=$(echo $TRAVIS_BRANCH | cut -d'v' -f 2).$TRAVIS_BUILD_NUMBER
sed -i -e 's/0\.0\.0/'"$VERSION_NUMBER"'/g' setup.py
echo "VERSION " $VERSION_NUMBER
