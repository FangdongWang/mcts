set -x

PDSH_ROOT=$PWD/.venv/pdsh
echo $PDSH_ROOT
tar xf ./dependency/pdsh-2.31.tar.gz -C ./dependency/
cd ./dependency/pdsh-pdsh-2.31/
./configure \
--prefix=${PDSH_ROOT} \
--with-ssh \
--with-machines=${PDSH_ROOT}/machines \
--with-dshgroups=${PDSH_ROOT}/group \
--with-rcmd-rank-list=ssh \
--with-exec && \
make && \
make install

cd ../..
echo $PWD
rm -rf ./dependency/pdsh-pdsh-2.31/