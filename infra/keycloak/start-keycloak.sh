#!/usr/bin/env bash
set -euo pipefail

export KC_DB_PASSWORD="$(< /run/secrets/keycloak_db_password)"
export KC_BOOTSTRAP_ADMIN_PASSWORD="$(< /run/secrets/keycloak_admin_password)"

exec /opt/keycloak/bin/kc.sh start \
  --optimized \
  --import-realm \
  --server-async-bootstrap=false
