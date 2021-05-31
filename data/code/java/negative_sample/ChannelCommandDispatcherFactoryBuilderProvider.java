/*
 * JBoss, Home of Professional Open Source.
 * Copyright 2014, Red Hat, Inc., and individual contributors
 * as indicated by the @author tags. See the copyright.txt file in the
 * distribution for a full listing of individual contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */
package org.wildfly.clustering.server.dispatcher;

import org.jboss.modules.ModuleIdentifier;
import org.wildfly.clustering.dispatcher.CommandDispatcherFactory;
import org.wildfly.clustering.server.GroupBuilderFactory;
import org.wildfly.clustering.service.Builder;
import org.wildfly.clustering.spi.DistributedGroupBuilderProvider;

/**
 * Provides the requisite builders for creating a channel-based {@link org.wildfly.clustering.dispatcher.CommandDispatcherFactory}.
 * @author Paul Ferraro
 */
public class ChannelCommandDispatcherFactoryBuilderProvider extends CommandDispatcherFactoryBuilderProvider implements DistributedGroupBuilderProvider {

    private static final GroupBuilderFactory<CommandDispatcherFactory> FACTORY = new GroupBuilderFactory<CommandDispatcherFactory>() {
        @Override
        public Builder<CommandDispatcherFactory> createBuilder(String group, ModuleIdentifier module) {
            return new ChannelCommandDispatcherFactoryBuilder(group, module);
        }
    };

    public ChannelCommandDispatcherFactoryBuilderProvider() {
        super(FACTORY);
    }
}
